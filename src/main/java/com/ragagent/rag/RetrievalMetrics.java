package com.ragagent.rag;

import java.util.List;

/**
 * 检索评测的指标计算。
 *
 * 为什么抽成独立的纯函数类：指标算错是最难发现的一类 bug —— 它不会抛异常，只会让你
 * 基于错误的数字做决策（以为调参涨了 5 个点，其实没变）。纯函数才能被单测钉住。
 *
 * 本项目评测集的标注是 0/1（每个问题只有一个期望文档），在这个前提下各指标的口径：
 *
 *   Recall@k / Hit@k —— 期望文档是否出现在前 k 名。单相关文档时两者完全等价。
 *   MRR              —— 1/rank，未命中记 0。比 Recall 多一层位置信息。
 *   nDCG@k           —— DCG/IDCG。DCG = Σ (2^rel - 1) / log2(i + 1)，
 *                       单相关文档、理想排序是它排第 1，所以 IDCG = 1，
 *                       于是 nDCG@k 退化成 1 / log2(rank + 1)。
 *                       加它主要是因为"nDCG 涨了多少"最常被拿来横比方案优劣，口径得对得上。
 */
public final class RetrievalMetrics {

    private RetrievalMetrics() {
    }

    /**
     * 期望文档在结果里的名次，从 1 开始；没找到返回 0。
     * 用 0 表示"未命中"而不是 -1，是为了让调用方可以直接用 rank &gt; 0 判断。
     */
    public static int rankOf(List<String> files, String expectedFile) {
        if (expectedFile == null || files == null) {
            return 0;
        }
        for (int i = 0; i < files.size(); i++) {
            if (expectedFile.equals(files.get(i))) {
                return i + 1;
            }
        }
        return 0;
    }

    /** rank 从 1 开始；未命中传 0 或负数，返回 0。 */
    public static double reciprocalRank(int rank) {
        return rank <= 0 ? 0.0 : 1.0 / rank;
    }

    /** 单相关文档 0/1 标注下的 nDCG@k：命中则 1/log2(rank+1)，未命中或超出窗口记 0。 */
    public static double ndcgAtK(int rank, int k) {
        if (rank <= 0 || rank > k) {
            return 0.0;
        }
        return 1.0 / (Math.log(rank + 1) / Math.log(2));
    }
}
