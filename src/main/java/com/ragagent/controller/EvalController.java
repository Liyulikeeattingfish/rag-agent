package com.ragagent.controller;

import com.ragagent.config.RetrievalProperties;
import com.ragagent.rag.RetrievedSegment;
import com.ragagent.service.DocumentService;
import com.ragagent.service.RetrievalService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 检索评测接口（RAG 的体检报告）。
 *
 * 为什么要有它：改了切块大小、换了 embedding 模型、调了 minScore / rerank 阈值之后，
 * 只靠"问几个问题、肉眼看着还行"，你根本说不出"到底变好了没有"。有固定用例集，
 * 每次改完跑一遍看指标变化，改动才是有依据的。
 *
 * 指标口径（这一版做了升级，从"关键词有没有命中"改为"文档有没有被排进 top3"）：
 *
 *   Recall@3  —— 期望文档出现在最终 top3 里的比例。衡量"能不能捞回来"。
 *   MRR       —— 期望文档排名的倒数的均值（没进 top3 记 0）。衡量"捞回来的位置好不好"。
 *                Recall 只看有没有，MRR 还能看出"从第 3 名提到第 1 名"这种进步，
 *                在语料很小、Recall 容易饱和的场景里，MRR 才是真正能分辨优劣的指标。
 *   拒答率    —— 无关用例中，最终上下文为空（系统拒答）的比例。衡量"会不会硬答"。
 *
 * 一个刻意的设计：评测【不调用大模型】。这三个指标全是"检索层能不能排对"，
 * 跟大模型会不会胡说八道是两件事。把 LLM 拿出去之后，跑一次评测只要几十次 embedding
 * 调用，几秒钟出结果，可以随便跑；掺进 LLM 就变成又慢又贵又不可重复了。
 *
 * 另一个设计：expectedFile 不在库里的用例标记为 skipped 并排除出分母，
 * 而不是算作 miss —— 那是"语料缺失"，不是"检索不行"，混在一起会掩盖真问题。
 *
 * 用法：GET /eval
 */
@RestController
public class EvalController {

    /**
     * 一条评测用例。
     *
     * @param query        问题
     * @param expectedFile 期望被排进 top3 的文档文件名；无关用例为 null
     * @param shouldAnswer true=语料里有答案；false=无关问题，期望系统拒答
     */
    private record EvalCase(String query, String expectedFile, boolean shouldAnswer) {
    }

    /**
     * 用例集（20 条，覆盖三类）：
     *   (a) 普通语义问题 —— 考察向量通道的语义泛化能力（问法和原文措辞不同）
     *   (b) 精确关键词问题 —— 考察关键词通道，这类问题里"某个 token 原样出现"才是硬条件，
     *       纯向量检索最容易在这里翻车，混合检索的价值也主要体现在这里
     *   (c) 无关问题 —— 考察拒答能力，防止"问什么都能答"
     *
     * 注意：expectedFile 必须与已入库的真实文件名完全一致。改名、删除文档之后
     * 这些用例会被自动跳过（返回体里的 skipped 字段会说明），不会静默变成假失败。
     */
    private static final List<EvalCase> CASES = List.of(
            // ---- (a) 普通语义问题 ----
            new EvalCase("我是哪个学校的", "my-note.txt", true),
            new EvalCase("我的项目用了什么技术栈", "my-note.txt", true),
            new EvalCase("我做过什么上线的项目", "my-note.txt", true),
            new EvalCase("这个微信小程序是干什么的", "my-note.txt", true),
            new EvalCase("RAG 是什么", "sample.txt", true),
            new EvalCase("文档切块大小怎么定", "sample.txt", true),
            new EvalCase("为什么向量维度必须保持一致", "sample.txt", true),
            new EvalCase("向量数据库和关系数据库有什么区别", "vector-db-notes-v2.md", true),
            new EvalCase("怎么在召回率和速度之间做权衡", "vector-db-notes-v2.md", true),
            new EvalCase("IVFFlat 和 HNSW 应该怎么选", "vector-db-notes-v2.md", true),
            new EvalCase("检索之后为什么还要做一层重排序", "vector-db-notes-v2.md", true),

            // ---- (b) 只有精确关键词才能命中的问题 ----
            new EvalCase("Qwen3-Embedding-0.6B 输出多少维", "sample.txt", true),
            new EvalCase("rag_segments 是什么表", "sample.txt", true),
            new EvalCase("vector_cosine_ops 是什么", "vector-db-notes-v2.md", true),
            new EvalCase("pgvector 的 probes 参数是干什么的", "vector-db-notes-v2.md", true),
            new EvalCase("精灵小窝是什么", "my-note.txt", true),

            // ---- (c) 无关问题（期望拒答） ----
            new EvalCase("今天天气怎么样", null, false),
            new EvalCase("推荐一部电影", null, false),
            new EvalCase("怎么炒西红柿鸡蛋", null, false),
            new EvalCase("Python 的 GIL 是什么", null, false)
    );

    private final RetrievalService retrievalService;
    private final DocumentService documentService;
    private final RetrievalProperties properties;

    public EvalController(RetrievalService retrievalService,
                          DocumentService documentService,
                          RetrievalProperties properties) {
        this.retrievalService = retrievalService;
        this.documentService = documentService;
        this.properties = properties;
    }

    @GetMapping("/eval")
    public Map<String, Object> eval() {
        Set<String> corpusFiles = documentService.listFileNames();

        int positiveTotal = 0;
        int negativeTotal = 0;
        int skipped = 0;
        int recallHit = 0;          // 期望文档进了 top3
        int top1Hit = 0;            // 期望文档就在第 1 名（顺带统计，看精排有没有把答案顶上去）
        double reciprocalRankSum = 0;
        int refused = 0;            // 无关用例中，系统在检索层就拒答了

        List<Map<String, Object>> details = new ArrayList<>(CASES.size());

        System.out.println("[eval] ===== 开始评测：共 " + CASES.size() + " 条 | hybrid="
                + properties.isHybridEnabled() + " rerank=" + properties.isRerankEnabled()
                + " candidateTopK=" + properties.getCandidateTopK()
                + " minScore=" + properties.getMinScore()
                + " rerankMinScore=" + properties.getRerankMinScore()
                + " | 语料文件=" + corpusFiles + " =====");

        for (EvalCase evalCase : CASES) {
            // 期望文档不在语料里 —— 这条用例无法评判，跳过并从分母中剔除
            boolean fileMissing = evalCase.shouldAnswer()
                    && (evalCase.expectedFile() == null || !corpusFiles.contains(evalCase.expectedFile()));

            List<RetrievedSegment> hits = retrievalService.retrieve(evalCase.query());
            List<String> files = hits.stream().map(RetrievedSegment::fileName).toList();

            Map<String, Object> detail = new LinkedHashMap<>();
            detail.put("query", evalCase.query());
            detail.put("expectedFile", evalCase.expectedFile());
            detail.put("shouldAnswer", evalCase.shouldAnswer());
            detail.put("returnedFiles", files);
            detail.put("returnedScores", hits.stream().map(EvalController::scoreOf).toList());

            if (fileMissing) {
                skipped++;
                detail.put("skipped", true);
                detail.put("reason", "期望文档 " + evalCase.expectedFile() + " 不在当前语料里");
                details.add(detail);
                System.out.println("[eval] SKIP | Q=" + evalCase.query()
                        + " | 期望文档 " + evalCase.expectedFile() + " 不在语料中");
                continue;
            }
            detail.put("skipped", false);

            if (evalCase.shouldAnswer()) {
                positiveTotal++;

                int rank = 0;
                for (int i = 0; i < files.size(); i++) {
                    if (evalCase.expectedFile().equals(files.get(i))) {
                        rank = i + 1;
                        break;
                    }
                }
                boolean hit = rank > 0;
                if (hit) {
                    recallHit++;
                }
                if (rank == 1) {
                    top1Hit++;
                }
                reciprocalRankSum += hit ? 1.0 / rank : 0.0;

                detail.put("rank", rank == 0 ? null : rank);
                detail.put("hit", hit);

                System.out.println("[eval] 正面 | Q=" + evalCase.query()
                        + " | 期望=" + evalCase.expectedFile()
                        + " | 实际=" + files
                        + " | rank=" + (rank == 0 ? "miss" : rank)
                        + " | " + (hit ? "✅" : "❌"));
            } else {
                negativeTotal++;
                boolean isRefused = hits.isEmpty();
                if (isRefused) {
                    refused++;
                }
                detail.put("refused", isRefused);

                System.out.println("[eval] 反面 | Q=" + evalCase.query()
                        + " | 实际=" + files
                        + " | " + (isRefused ? "已拒答 ✅" : "被硬答 ❌"));
            }

            details.add(detail);
        }

        double recallAt3 = positiveTotal == 0 ? 0 : (double) recallHit / positiveTotal;
        double recallAt1 = positiveTotal == 0 ? 0 : (double) top1Hit / positiveTotal;
        double mrr = positiveTotal == 0 ? 0 : reciprocalRankSum / positiveTotal;
        double refusalRate = negativeTotal == 0 ? 0 : (double) refused / negativeTotal;

        System.out.println("[eval] ===== 完成：Recall@3=" + pct(recallAt3)
                + " Recall@1=" + pct(recallAt1)
                + " MRR=" + String.format("%.4f", mrr)
                + " 拒答率=" + pct(refusalRate)
                + " | 正面 " + positiveTotal + " 条，反面 " + negativeTotal
                + " 条，跳过 " + skipped + " 条 =====");

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("total", CASES.size());
        result.put("skipped", skipped);
        result.put("corpusFiles", corpusFiles);

        result.put("positiveTotal", positiveTotal);
        result.put("negativeTotal", negativeTotal);

        result.put("recallAt3", round4(recallAt3));
        result.put("recallAt1", round4(recallAt1));
        result.put("mrr", round4(mrr));
        result.put("refusalRate", round4(refusalRate));

        result.put("recallAt3Text", pct(recallAt3));
        result.put("recallAt1Text", pct(recallAt1));
        result.put("refusalRateText", pct(refusalRate));

        // 把生效的配置回显出来，否则两个不同的数字看起来会完全无法解释
        Map<String, Object> config = new LinkedHashMap<>();
        config.put("hybridEnabled", properties.isHybridEnabled());
        config.put("rerankEnabled", properties.isRerankEnabled());
        config.put("candidateTopK", properties.getCandidateTopK());
        config.put("finalTopK", properties.getFinalTopK());
        config.put("minScore", properties.getMinScore());
        config.put("keywordMinSimilarity", properties.getKeywordMinSimilarity());
        config.put("rrfK", properties.getRrfK());
        config.put("rerankMinScore", properties.getRerankMinScore());
        result.put("config", config);

        result.put("details", details);
        return result;
    }

    /** 最终分数：优先 rerank 分，其次 RRF 分，再次向量分。仅用于人工观察。 */
    private static Double scoreOf(RetrievedSegment segment) {
        if (segment.rerankScore() != null) {
            return round4(segment.rerankScore());
        }
        if (segment.rrfScore() != null) {
            return round4(segment.rrfScore());
        }
        return round4(segment.vectorScore());
    }

    private static Double round4(Double value) {
        return value == null ? null : Math.round(value * 10000) / 10000.0;
    }

    private static double round4(double value) {
        return Math.round(value * 10000) / 10000.0;
    }

    private static String pct(double ratio) {
        return String.format("%.1f%%", ratio * 100);
    }
}
