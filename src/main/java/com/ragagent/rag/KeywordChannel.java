package com.ragagent.rag;

import java.util.List;

/**
 * 关键词召回通道的统一抽象。
 *
 * 目前两个实现：
 *   TrigramKeywordSearcher —— PostgreSQL 的 pg_trgm，字符三元组相似度，查库
 *   Bm25KeywordSearcher    —— 内存里的 BM25 倒排索引，自己算 IDF 与长度归一化
 *
 * 用接口而不是 if-else 的意义在于：融合、去重、排序这些逻辑是通道无关的，
 * 加一条通道不该动 RetrievalService 的一行代码。
 */
public interface KeywordChannel {

    /**
     * @param query   用户问题原文
     * @param limit   取几条
     * @param minScore 分数下限。注意它不是跨通道可比的绝对值：
     *                 pg_trgm 的相似度是 0~1，BM25 的分数没有上界，硬套同一个阈值没有意义 ——
     *                 所以 BM25 实现只做"至少命中一个查询词"的判断，忽略这个参数。
     */
    List<KeywordHit> search(String query, int limit, double minScore);

    /**
     * 语料发生变化后作废内部缓存。pg_trgm 每次直接查库，不需要实现这个方法；
     * BM25 需要重建倒排索引，否则新入库的文档永远搜不到。
     */
    default void invalidate() {
    }
}
