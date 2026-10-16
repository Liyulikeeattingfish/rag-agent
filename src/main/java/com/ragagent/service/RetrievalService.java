package com.ragagent.service;

import com.ragagent.config.RetrievalProperties;
import com.ragagent.rag.KeywordSearcher;
import com.ragagent.rag.RagMetadataKeys;
import com.ragagent.rag.RetrievedSegment;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 检索编排：把「向量召回」「关键词召回」「RRF 融合」「rerank 精排」串成一条流水线。
 *
 * 完整链路（默认配置）：
 *
 *   问题 ──┬─→ 向量通道 top10（EmbeddingStore.search，受 min-score 约束）
 *          └─→ 关键词通道 top10（pg_trgm similarity）
 *                    │
 *                    ▼
 *              RRF 融合去重 → 候选 10 条
 *                    │
 *                    ▼
 *              bge-reranker 精排 → 按 rerank 分排序
 *                    │
 *                    ▼
 *              过 rerank-min-score 闸门 → 取 finalTopK=3 → 喂给大模型
 *
 * 为什么要 RRF 而不是把两路分数加权相加（这是个高频面试题）：
 *   向量分是 (1+cos)/2 ∈ [0,1]，trigram 相似度也是 [0,1]，看着量纲一样，实际分布完全不同——
 *   前者不相关也有 0.6，后者不相关直接是 0。直接相加的话，权重就没法调了：
 *   调大一点向量分就压倒一切，调小一点关键词分就压倒一切，中间没有稳定区间。
 *   RRF 只取「名次」不取「分数」：score(d) = Σ 1/(k + rank)。名次是无量纲的，
 *   两个通道天然可比，也就彻底摆脱了调权重的泥潭。代价是丢掉了分数的绝对大小信息
 *   （第 1 名和第 2 名差多少，RRF 不关心）——这个损失由后面的 rerank 补回来。
 *
 * 三段可独立开关（见 RetrievalProperties），用于做消融对比：
 *   hybrid=false + rerank=false → 纯向量检索（改造前的行为，作为对比基线）
 *   hybrid=true  + rerank=false → 混合召回 + RRF
 *   hybrid=true  + rerank=true  → 混合召回 + RRF + 精排（默认）
 */
@Service
public class RetrievalService {

    private static final Logger log = LoggerFactory.getLogger(RetrievalService.class);

    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> embeddingStore;
    private final KeywordSearcher keywordSearcher;
    private final RerankClient rerankClient;
    private final RetrievalProperties properties;

    public RetrievalService(EmbeddingModel embeddingModel,
                            EmbeddingStore<TextSegment> embeddingStore,
                            KeywordSearcher keywordSearcher,
                            RerankClient rerankClient,
                            RetrievalProperties properties) {
        this.embeddingModel = embeddingModel;
        this.embeddingStore = embeddingStore;
        this.keywordSearcher = keywordSearcher;
        this.rerankClient = rerankClient;
        this.properties = properties;
    }

    /**
     * 检索出最终要喂给大模型的片段（已排序、已截断到 finalTopK）。
     * 返回空列表表示"检索不到够格的资料"，调用方应当直接拒答。
     */
    public List<RetrievedSegment> retrieve(String question) {
        // 两个开关都关掉 = 改造前的纯向量检索，专门留给"改造前后对比"用
        if (!properties.isHybridEnabled() && !properties.isRerankEnabled()) {
            return baselineVectorOnly(question);
        }

        List<Candidate> candidates = coarseRecall(question);
        if (candidates.isEmpty()) {
            log.info("[retrieval] Q={} 粗召回为空（两个通道都没有过闸的候选）", question);
            return List.of();
        }

        boolean reranked = false;
        if (properties.isRerankEnabled()) {
            reranked = applyRerank(question, candidates);
        }

        List<RetrievedSegment> result = new ArrayList<>();
        for (Candidate candidate : candidates) {
            // 过 rerank 闸门。只有真做了 rerank 才谈得上 rerank 阈值；
            // 若 rerank 调用失败降级了，候选上的 rerankScore 是 null，此时不能拿它去过滤。
            if (reranked && candidate.rerankScore != null
                    && candidate.rerankScore < properties.getRerankMinScore()) {
                continue;
            }
            result.add(candidate.toSegment());
            if (result.size() >= properties.getFinalTopK()) {
                break;
            }
        }

        logRetrieval(question, candidates, result);
        return result;
    }

    // ------------------------------------------------------------------
    // 第 1 段：粗召回（两个通道并行取回，再由 RRF 融合）
    // ------------------------------------------------------------------

    private List<Candidate> coarseRecall(String question) {
        List<Candidate> vectorHits = vectorChannel(question);

        List<Candidate> keywordHits = List.of();
        if (properties.isHybridEnabled()) {
            keywordHits = keywordChannel(question);
        }

        List<Candidate> fused = fuseByRrf(vectorHits, keywordHits);

        log.info("[retrieval] Q={} 粗召回：向量 {} 条 / 关键词 {} 条 → RRF 融合后取 {} 条候选",
                question, vectorHits.size(), keywordHits.size(), fused.size());
        return fused;
    }

    /**
     * 向量通道。
     *
     * 这里给 search 带了 minScore 闸门——这是有意为之，不是为了省事：
     * 粗召回如果完全不设闸，问「今天天气」也会稳稳返回 10 条候选，
     * 后面的 rerank 再把它们的分数压到 0.000001 也来不及了（因为 rerank 只排序、不拒答）。
     * 也就是说，「拒答」这件事必须在粗召回就有一道闸，不能全指望后面的精排。
     */
    private List<Candidate> vectorChannel(String question) {
        Embedding queryEmbedding = embeddingModel.embed(question).content();

        EmbeddingSearchResult<TextSegment> searchResult = embeddingStore.search(
                EmbeddingSearchRequest.builder()
                        .queryEmbedding(queryEmbedding)
                        .maxResults(properties.getVectorTopK())
                        .minScore(properties.getMinScore())
                        .build());

        List<Candidate> hits = new ArrayList<>();
        for (EmbeddingMatch<TextSegment> match : searchResult.matches()) {
            TextSegment segment = match.embedded();
            Candidate candidate = new Candidate(
                    segment.text(),
                    segment.metadata().getString(RagMetadataKeys.FILE_NAME),
                    segment.metadata().getInteger(RagMetadataKeys.SEGMENT_INDEX));
            candidate.vectorScore = match.score();
            hits.add(candidate);
        }
        return hits;
    }

    private List<Candidate> keywordChannel(String question) {
        List<KeywordSearcher.KeywordHit> hits = keywordSearcher.search(
                question, properties.getKeywordTopK(), properties.getKeywordMinSimilarity());

        List<Candidate> candidates = new ArrayList<>(hits.size());
        for (KeywordSearcher.KeywordHit hit : hits) {
            Candidate candidate = new Candidate(hit.text(), hit.fileName(), hit.segmentIndex());
            candidate.keywordScore = hit.similarity();
            candidates.add(candidate);
        }
        return candidates;
    }

    /**
     * RRF 融合。
     *
     * 关键点：同一段可能被两个通道同时召回，这时要合并成一条（去重），
     * 并把它的 RRF 分累加——被两个通道都认可的段落，分数天然更高，这正是融合想要的效果。
     * 去重的身份用「文件名 + 片段序号」，而不是用文本内容：
     * 两份不同文档可能切出一模一样的话，用内容去重会把它们错误合并成一段。
     */
    private List<Candidate> fuseByRrf(List<Candidate> vectorHits, List<Candidate> keywordHits) {
        Map<String, Candidate> merged = new LinkedHashMap<>();

        for (Candidate candidate : vectorHits) {
            merged.putIfAbsent(identityKey(candidate), candidate);
        }
        for (Candidate candidate : keywordHits) {
            Candidate existing = merged.get(identityKey(candidate));
            if (existing == null) {
                merged.put(identityKey(candidate), candidate);
            } else {
                // 同一个片段被关键词通道也命中了：把分数补上，但对象仍以先到的（向量那条）为准
                existing.keywordScore = candidate.keywordScore;
            }
        }

        accumulateRrf(vectorHits, merged);
        accumulateRrf(keywordHits, merged);

        List<Candidate> fused = new ArrayList<>(merged.values());
        fused.sort(Comparator.comparingDouble((Candidate c) -> c.rrfScore).reversed());

        return fused.size() <= properties.getCandidateTopK()
                ? fused
                : new ArrayList<>(fused.subList(0, properties.getCandidateTopK()));
    }

    private void accumulateRrf(List<Candidate> ranked, Map<String, Candidate> merged) {
        for (int i = 0; i < ranked.size(); i++) {
            Candidate canonical = merged.get(identityKey(ranked.get(i)));
            if (canonical != null) {
                canonical.rrfScore += 1.0 / (properties.getRrfK() + i + 1);
            }
        }
    }

    /** 片段身份：文件名 + 片段序号；两者缺失时退化成用原文哈希兜底。 */
    private static String identityKey(Candidate candidate) {
        if (candidate.fileName != null && candidate.segmentIndex != null) {
            return candidate.fileName + "#" + candidate.segmentIndex;
        }
        return "text#" + candidate.text.hashCode();
    }

    // ------------------------------------------------------------------
    // 第 2 段：rerank 精排
    // ------------------------------------------------------------------

    /**
     * 对候选做精排。
     *
     * @return 是否真的精排成功。失败时返回 false，候选保持 RRF 顺序——
     *         外部服务抖动不应该让 /ask 直接 500，但它必须被记成 ERROR 并在返回体里可见。
     */
    private boolean applyRerank(String question, List<Candidate> candidates) {
        try {
            List<String> documents = candidates.stream().map(c -> c.text).toList();
            Map<Integer, Double> scores = rerankClient.rerank(question, documents);

            for (int i = 0; i < candidates.size(); i++) {
                candidates.get(i).rerankScore = scores.get(i);
            }
            candidates.sort(Comparator.comparingDouble(
                    (Candidate c) -> c.rerankScore == null ? -1.0 : c.rerankScore).reversed());
            return true;
        } catch (Exception e) {
            log.error("[retrieval] Q={} rerank 调用失败，已降级为 RRF 排序（本次不精排）：{}",
                    question, e.toString());
            return false;
        }
    }

    // ------------------------------------------------------------------
    // 改造前的基线路径：纯向量 topN + min-score 闸门
    // ------------------------------------------------------------------

    private List<RetrievedSegment> baselineVectorOnly(String question) {
        Embedding queryEmbedding = embeddingModel.embed(question).content();

        EmbeddingSearchResult<TextSegment> searchResult = embeddingStore.search(
                EmbeddingSearchRequest.builder()
                        .queryEmbedding(queryEmbedding)
                        .maxResults(properties.getFinalTopK())
                        .minScore(properties.getMinScore())
                        .build());

        List<RetrievedSegment> result = new ArrayList<>();
        for (EmbeddingMatch<TextSegment> match : searchResult.matches()) {
            TextSegment segment = match.embedded();
            result.add(new RetrievedSegment(
                    segment.text(),
                    segment.metadata().getString(RagMetadataKeys.FILE_NAME),
                    segment.metadata().getInteger(RagMetadataKeys.SEGMENT_INDEX),
                    match.score(),
                    null,
                    null,
                    null));
        }
        log.info("[retrieval] Q={} 基线模式（纯向量）：{} 条", question, result.size());
        return result;
    }

    // ------------------------------------------------------------------

    private void logRetrieval(String question, List<Candidate> candidates, List<RetrievedSegment> result) {
        StringBuilder sb = new StringBuilder();
        for (RetrievedSegment segment : result) {
            sb.append("\n    - ").append(segment.fileName())
                    .append("#").append(segment.segmentIndex())
                    .append(" vector=").append(fmt(segment.vectorScore()))
                    .append(" keyword=").append(fmt(segment.keywordScore()))
                    .append(" rrf=").append(fmt(segment.rrfScore()))
                    .append(" rerank=").append(fmt(segment.rerankScore()));
        }
        log.info("[retrieval] Q={} 候选 {} 条 → 最终 {} 条{}",
                question, candidates.size(), result.size(), sb);
    }

    private static String fmt(Double value) {
        return value == null ? "-" : String.format("%.4f", value);
    }

    /**
     * 检索过程中的可变候选对象。
     * 做成可变是因为它要在三个通道/阶段来回累加分数（向量分、关键词分、RRF 分、rerank 分），
     * 每步都 new 一个新对象只会让代码更难读。对外只暴露不可变的 RetrievedSegment。
     */
    private static final class Candidate {
        final String text;
        final String fileName;
        final Integer segmentIndex;
        Double vectorScore;
        Double keywordScore;
        double rrfScore;
        Double rerankScore;

        Candidate(String text, String fileName, Integer segmentIndex) {
            this.text = text;
            this.fileName = fileName;
            this.segmentIndex = segmentIndex;
        }

        RetrievedSegment toSegment() {
            return new RetrievedSegment(text, fileName, segmentIndex,
                    vectorScore, keywordScore, rrfScore, rerankScore);
        }
    }
}
