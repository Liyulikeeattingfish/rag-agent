package com.ragagent.service;

import com.ragagent.rag.RagMetadataKeys;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.filter.MetadataFilterBuilder;
import dev.langchain4j.store.embedding.EmbeddingStore;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 文档域的领域逻辑：入库、查重、按文档删除、列出文档。
 *
 * 为什么把它从 Controller 里抽出来：Controller 应该只负责"HTTP 协议"那点事（解析参数、校验格式、
 * 拼返回体），而"查重规则、冲突怎么判、切多大的块"属于业务规则。两者混在一个类里的后果是——
 * 以后要加一个命令行导入、或者加个定时任务，就得去 new 一个 Controller，很别扭。
 */
@Service
public class DocumentService {

    /** 表名。必须与 EmbeddingConfig 里创建 EmbeddingStore 时指定的表名一致，否则这里查不到数据。 */
    private static final String TABLE = "rag_segments";

    // 切块大小（字符数）与相邻块的重叠（字符数）
    private static final int MAX_SEGMENT_SIZE = 500;
    private static final int MAX_OVERLAP_SIZE = 100;

    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> embeddingStore;
    private final JdbcTemplate jdbcTemplate;

    public DocumentService(EmbeddingModel embeddingModel,
                           EmbeddingStore<TextSegment> embeddingStore,
                           JdbcTemplate jdbcTemplate) {
        this.embeddingModel = embeddingModel;
        this.embeddingStore = embeddingStore;
        this.jdbcTemplate = jdbcTemplate;
    }

    /**
     * 入库一份文档：切块 → 向量化 → 写入 pgvector。入库前做两级判定。
     *
     * 四种结局（返回体的 status 字段）：
     *   created   —— 全新文档，正常入库
     *   duplicate —— 库里已有完全相同的内容，跳过（与文件名无关）
     *   conflict  —— 同名文档已存在但内容不同，默认拒绝，提示可用 overwrite=true 覆盖
     *   updated   —— 同名新版本，已先删掉旧片段再入库
     *
     * @param fileName  文件名，同时作为该文档的身份标识
     * @param text      文档全文
     * @param overwrite 遇到"同名不同内容"时是否覆盖旧版本
     */
    public Map<String, Object> ingest(String fileName, String text, boolean overwrite) {
        String contentHash = sha256Hex(text);

        // 第 1 级：内容级去重。库里已有同样内容就跳过，与文件名无关——所以「改个名字重传」也认得出来。
        if (countByContentHash(contentHash) > 0) {
            Map<String, Object> r = new LinkedHashMap<>();
            r.put("status", "duplicate");
            r.put("message", "该文档内容已存在，跳过入库");
            r.put("filename", fileName);
            r.put("contentHash", contentHash);
            r.put("success", false);
            return r;
        }

        // 第 2 级：文档级冲突。同名文档已在库里，但内容不同 → 视为"这份文档被修改过"，即新版本。
        // 这里默认【拒绝】而不是默默并存：并存会让检索时新旧内容同时被召回，直接污染答案。
        // 但也不自动覆盖——覆盖是破坏性操作，必须由调用方显式同意（overwrite=true）。
        int existingSegments = countByFileName(fileName);
        boolean isNewVersion = existingSegments > 0;
        if (isNewVersion && !overwrite) {
            Map<String, Object> r = new LinkedHashMap<>();
            r.put("status", "conflict");
            r.put("message", "同名文档已存在且内容不同（视为该文档的新版本）。"
                    + "如需覆盖旧版本，请重新上传并加上参数 overwrite=true");
            r.put("filename", fileName);
            r.put("existingSegments", existingSegments);
            r.put("success", false);
            return r;
        }

        // 是覆盖场景就先删干净旧版本，再写新的，避免新旧片段并存
        int removed = isNewVersion ? deleteByFileName(fileName) : 0;

        // 把 file_name / content_hash 放进 Document 的 metadata。
        // 关键机制：切分器生成每个片段时会复制 Document 的 metadata（createSegment(String, Document, int)），
        // 所以我们只需要在 Document 上写一次，每一块都会自动带上这两个字段。
        Metadata metadata = new Metadata()
                .put(RagMetadataKeys.FILE_NAME, fileName)
                .put(RagMetadataKeys.CONTENT_HASH, contentHash);
        Document document = Document.from(text, metadata);
        DocumentSplitter splitter = DocumentSplitters.recursive(MAX_SEGMENT_SIZE, MAX_OVERLAP_SIZE);
        List<TextSegment> segments = splitter.split(document);

        // 调硅基流动 embedding 转向量，然后写入 pgvector
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        List<String> ids = embeddingStore.addAll(embeddings, segments);

        Map<String, Object> r = new LinkedHashMap<>();
        r.put("status", isNewVersion ? "updated" : "created");
        r.put("filename", fileName);
        r.put("contentHash", contentHash);
        r.put("segments", segments.size());
        r.put("saved", ids.size());
        if (isNewVersion) {
            r.put("removed", removed);   // 覆盖场景下顺带告诉调用方淘汰了多少旧片段
        }
        r.put("success", true);
        return r;
    }

    /** 列出库里已有的文档，按 文件名 + 内容哈希 聚合。历史数据没有 file_name 的会归到「(未命名)」。 */
    public Map<String, Object> listDocuments() {
        String sql = """
                SELECT COALESCE(metadata->>'file_name', '(未命名)') AS file_name,
                       metadata->>'content_hash'                   AS content_hash,
                       COUNT(*)                                    AS segments
                FROM %s
                GROUP BY 1, 2
                ORDER BY 1
                """.formatted(TABLE);

        List<Map<String, Object>> documents = jdbcTemplate.query(sql, (rs, rowNum) -> {
            Map<String, Object> doc = new LinkedHashMap<>();
            doc.put("filename", rs.getString("file_name"));
            doc.put("contentHash", rs.getString("content_hash"));
            doc.put("segments", rs.getInt("segments"));
            return doc;
        });
        int totalSegments = documents.stream().mapToInt(d -> (Integer) d.get("segments")).sum();

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("count", documents.size());
        result.put("totalSegments", totalSegments);
        result.put("documents", documents);
        return result;
    }

    /**
     * 库里出现过的所有文件名（不含历史遗留的无名数据）。
     *
     * 用途：评测时先确认"用例期望命中的那份文档到底在不在库里"。
     * 不在 → 这条用例应当被标记为 skipped 而不是 miss，否则评测结果里会混进
     * 一堆"语料缺失"造成的假失败，把真正的检索问题掩盖掉。
     */
    public Set<String> listFileNames() {
        List<String> names = jdbcTemplate.queryForList(
                "SELECT DISTINCT metadata->>'file_name' FROM " + TABLE
                        + " WHERE metadata->>'file_name' IS NOT NULL",
                String.class);
        return new LinkedHashSet<>(names);
    }

    /**
     * 删除某个文档的全部片段，返回被删掉的段数（文档不存在返回 0）。
     *
     * 这里走 EmbeddingStore.removeAll(Filter) 这个抽象，而不是直接写 DELETE SQL：
     * 好处是以后把向量库换成 Milvus / Qdrant，这个方法一行都不用改。
     * 代价是有个前置条件——必须确认该实现真的支持按元数据过滤删除（有些实现只抛 UnsupportedOperationException）。
     * 本项目已核对过 langchain4j-pgvector:1.0.0-beta5 实现了它，生成的条件形如 (metadata->>'file_name')::text = ?。
     */
    public int deleteByFileName(String fileName) {
        int before = countByFileName(fileName);
        if (before == 0) {
            return 0;
        }
        embeddingStore.removeAll(
                MetadataFilterBuilder.metadataKey(RagMetadataKeys.FILE_NAME).isEqualTo(fileName));
        return before;
    }

    /** 内容哈希是否已入库。 */
    private int countByContentHash(String contentHash) {
        Integer count = jdbcTemplate.queryForObject(
                "SELECT COUNT(*) FROM " + TABLE + " WHERE metadata->>'content_hash' = ?",
                Integer.class, contentHash);
        return count == null ? 0 : count;
    }

    /** 同名文档当前有多少段（0 表示这份文档还没入库过）。 */
    private int countByFileName(String fileName) {
        Integer count = jdbcTemplate.queryForObject(
                "SELECT COUNT(*) FROM " + TABLE + " WHERE metadata->>'file_name' = ?",
                Integer.class, fileName);
        return count == null ? 0 : count;
    }

    /** 整份文本内容的 SHA-256，转成小写十六进制字符串。 */
    private static String sha256Hex(String text) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] bytes = digest.digest(text.getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder(bytes.length * 2);
            for (byte b : bytes) {
                sb.append(Character.forDigit((b >> 4) & 0xF, 16));
                sb.append(Character.forDigit(b & 0xF, 16));
            }
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 算法不可用", e);
        }
    }
}
