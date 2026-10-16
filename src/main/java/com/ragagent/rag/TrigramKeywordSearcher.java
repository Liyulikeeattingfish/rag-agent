package com.ragagent.rag;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * 关键词通道之一：pg_trgm 三元组相似度。
 *
 * 为什么在有了向量检索之后还要这一路：
 * 向量检索擅长"语义相近"，但恰恰不擅长"字面精确相等"。问「vector_cosine_ops 是什么」、
 * 「Qwen3-Embedding-0.6B 多少维」、「500 报错怎么解决」这类问题时，答案要求的是
 * 【这个词原样出现过】，而 embedding 会把这个专有名词和一堆语义相近的词平均到一起，
 * 反而可能排不过一句语义相关但不含该词的话。关键词通道补的就是这个短板。
 *
 * 关于那个 GIN 索引，有一句实话要说清楚（这里最容易踩坑）：
 *   gin (text gin_trgm_ops) 主要加速的是「相似度阈值过滤」这条路，也就是 % 运算符
 *   （它读的是 pg_trgm.similarity_threshold 这个 GUC）。而本类用的是
 *   similarity(text, ?) >= ? 这种"算出分数再比较"的写法 —— 它能得到精确可控的分数，
 *   但过滤条件对索引不友好，走的是顺序扫描。
 *   本项目语料只有几十个片段，顺序扫描完全够用；真到百万级片段时，正确做法是改成
 *   WHERE text % ? ORDER BY similarity(text, ?) DESC LIMIT ?
 *   并给会话设 SET LOCAL pg_trgm.similarity_threshold = 0 —— 这样 GIN 索引才会真正被用上。
 *
 * 上线方式：rag.retrieval.keyword-channel 缺省或等于 trigram 时启用本实现。
 */
@Component
@ConditionalOnProperty(prefix = "rag.retrieval", name = "keyword-channel",
        havingValue = "trigram", matchIfMissing = true)
public class TrigramKeywordSearcher implements KeywordChannel {

    /** 与 EmbeddingConfig 建表、DocumentService 查重用的表名保持一致。 */
    private static final String TABLE = "rag_segments";

    private final JdbcTemplate jdbcTemplate;

    public TrigramKeywordSearcher(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    /**
     * 按 trigram 相似度降序取 top K。
     *
     * @param query         用户问题原文
     * @param limit         取几条
     * @param minSimilarity 相似度下限（严格大于）；0.0 表示只要求有交集
     */
    @Override
    public List<KeywordHit> search(String query, int limit, double minSimilarity) {
        String sql = """
                SELECT text                                  AS text,
                       metadata->>'file_name'                AS file_name,
                       NULLIF(metadata->>'index', '')::int   AS segment_index,
                       similarity(text, ?)                   AS score
                FROM %s
                WHERE similarity(text, ?) > ?
                ORDER BY score DESC
                LIMIT ?
                """.formatted(TABLE);

        return jdbcTemplate.query(sql, (rs, rowNum) -> new KeywordHit(
                rs.getString("text"),
                rs.getString("file_name"),
                (Integer) rs.getObject("segment_index"),
                rs.getDouble("score")
        ), query, query, minSimilarity, limit);
    }
}
