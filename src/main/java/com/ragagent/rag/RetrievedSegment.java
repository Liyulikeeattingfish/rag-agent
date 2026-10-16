package com.ragagent.rag;

/**
 * 一条「跑完全部检索流程」的候选片段。
 *
 * 为什么把所有分数都留下来，而不是只留一个最终分：
 * 排查检索质量问题时，你必须知道"它为什么排在这里"。只留一个融合后的数字，
 * 结果就是"分低了但不知道为什么低"。四路分数各行其责：
 *
 *   vectorScore  —— 向量通道的原始分数（LangChain4j pgvector 的 score = (1+cos)/2）
 *   keywordScore —— pg_trgm 通道的 trigram 相似度
 *   rrfScore     —— 两路按排名融合后的分数（只由名次决定，与原始分数量纲无关）
 *   rerankScore  —— bge-reranker 精排分数（0~1，语义相关性）
 *
 * 任意一路没参与，对应字段就是 null —— 比如纯关键词命中的片段没有 vectorScore，
 * 关掉 rerank 时所有片段都没有 rerankScore。null 比填 0 更诚实：0 是"算出来是 0"，null 是"没参与"。
 */
public record RetrievedSegment(
        String text,
        String fileName,
        Integer segmentIndex,
        Double vectorScore,
        Double keywordScore,
        Double rrfScore,
        Double rerankScore) {
}
