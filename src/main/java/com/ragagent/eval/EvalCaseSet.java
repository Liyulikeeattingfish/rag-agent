package com.ragagent.eval;

import java.util.List;

/**
 * 评测用例集（47 条 = 42 正例 + 5 无关）。
 *
 * 为什么单独抽成一个类：检索侧评测（/eval）和生成侧评测（/eval-answer）必须跑同一套用例，
 * 否则两边的数字对不上，也没法解释"检索对了但答案不忠实"这种组合结论。
 *
 * 语料是 docs/corpus/ 下的 11 份技术文档 + 项目根目录的 my-note.txt，共 12 份 22 段，
 * 每份文档覆盖一个独立主题、彼此不重叠 —— 因为"expectedFile 唯一"是这套指标成立的前提：
 * 如果两份文档都能回答同一个问题，期望文档写哪一个都是在冤枉检索。
 *
 * expectedFile 必须与已入库的真实文件名完全一致。改名、删除文档之后这些用例会被自动跳过
 * （返回体里的 skipped 字段会说明），不会静默变成假失败。
 */
public final class EvalCaseSet {

    public static final List<EvalCase> CASES = List.of(
            // ---- (a) 用户笔记 ----
            new EvalCase("我是哪个学校的", "my-note.txt", true),
            new EvalCase("我的项目用了什么技术栈", "my-note.txt", true),
            new EvalCase("我做过什么上线的项目", "my-note.txt", true),

            // ---- (b) 语义改写：问法与原文措辞不同 ----
            new EvalCase("RAG 是为了解决什么问题才被提出来的", "rag-overview.md", true),
            new EvalCase("RAG 分哪两个阶段，各自做什么", "rag-overview.md", true),
            new EvalCase("为什么不能把整篇文档直接塞进提示词", "rag-overview.md", true),
            new EvalCase("一份文档为什么要拆成小块再入库", "chunking.md", true),
            new EvalCase("相邻文本块之间留重叠部分是为了什么", "chunking.md", true),
            new EvalCase("按分隔符递归切分的降级顺序是怎样的", "chunking.md", true),
            new EvalCase("小块检索命中后返回大块给模型，这种方案叫什么", "chunking.md", true),
            new EvalCase("问题和文档分别独立编码有什么好处和代价", "embedding.md", true),
            new EvalCase("换了向量模型之后为什么必须重建表", "embedding.md", true),
            new EvalCase("向量做归一化之后余弦相似度等价于什么运算", "embedding.md", true),
            new EvalCase("分层图索引上层节点少是为了什么", "hnsw.md", true),
            new EvalCase("HNSW 的三个参数分别管什么", "hnsw.md", true),
            new EvalCase("哪个参数可以在线调整而不用重建索引", "hnsw.md", true),
            new EvalCase("IVFFlat 的簇数量应该怎么估计", "ivfflat.md", true),
            new EvalCase("为什么聚类索引必须先把数据灌进去再建", "ivfflat.md", true),
            new EvalCase("BM25 里的 IDF 起什么作用", "bm25.md", true),
            new EvalCase("BM25 的 k1 和 b 两个参数各自控制什么", "bm25.md", true),
            new EvalCase("中文场景下 BM25 为什么需要先分词", "bm25.md", true),
            new EvalCase("混合检索里两路分数为什么不能直接加权相加", "hybrid-rrf.md", true),
            new EvalCase("RRF 的常数 k 取大取小有什么区别", "hybrid-rrf.md", true),
            new EvalCase("关键词通道补的是向量检索的哪个短板", "hybrid-rrf.md", true),
            new EvalCase("双塔和交叉编码器的本质区别是什么", "rerank.md", true),
            new EvalCase("重排模型为什么只能对少量候选使用", "rerank.md", true),
            new EvalCase("重排模型会不会自己拒绝回答不相关的问题", "rerank.md", true),
            new EvalCase("nDCG 这个指标是怎么计算出来的", "rag-eval.md", true),
            new EvalCase("MRR 比召回率多提供了什么信息", "rag-eval.md", true),
            new EvalCase("faithfulness 这个指标衡量的是什么", "rag-eval.md", true),
            new EvalCase("检索侧评测和生成侧评测为什么要拆成两套", "rag-eval.md", true),
            new EvalCase("幻觉产生的根源到底是什么", "grounding.md", true),
            new EvalCase("引用溯源为什么需要规则校验和模型校验两层", "grounding.md", true),
            new EvalCase("拒答阈值定高和定低分别会犯什么错", "grounding.md", true),

            // ---- (c) 精确关键词：必须原样命中某个 token ----
            new EvalCase("ef_search 是什么参数", "hnsw.md", true),
            new EvalCase("vector_cosine_ops 是什么", "pgvector.md", true),
            new EvalCase("vector_ip_ops 对应哪个操作符", "pgvector.md", true),
            new EvalCase("vector 类型单列最多支持多少维", "pgvector.md", true),
            new EvalCase("查询的排序操作符和索引的 ops 类不匹配会怎样", "pgvector.md", true),
            new EvalCase("返回结果里的 index 字段是干什么的", "rerank.md", true),
            new EvalCase("Qwen3-Embedding-0.6B 输出多少维", "embedding.md", true),
            new EvalCase("精灵小窝是什么", "my-note.txt", true),

            // ---- (d) 无关问题（期望拒答） ----
            new EvalCase("今天天气怎么样", null, false),
            new EvalCase("推荐一部电影", null, false),
            new EvalCase("怎么炒西红柿鸡蛋", null, false),
            new EvalCase("Python 的 GIL 是什么", null, false),
            new EvalCase("帮我写一首关于春天的诗", null, false)
    );

    private EvalCaseSet() {
    }
}
