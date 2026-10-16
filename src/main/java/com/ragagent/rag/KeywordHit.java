package com.ragagent.rag;

/**
 * 一条关键词通道的命中结果。
 *
 * 为什么从 KeywordSearcher 内部提出来：现在有两个实现（pg_trgm 与 BM25），
 * 返回值必须是共用类型，否则 RetrievalService 得按实现分支处理，融合逻辑就脏了。
 *
 * 字段名用 score 而不是 similarity：pg_trgm 给的是相似度（0~1），BM25 给的是打分（没有上界），
 * 两者都只是"这个通道认为它有多相关"，叫 similarity 会误导。
 */
public record KeywordHit(String text, String fileName, Integer segmentIndex, double score) {
}
