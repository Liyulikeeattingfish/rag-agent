package com.ragagent.service;

import com.ragagent.eval.EvalCase;
import com.ragagent.eval.EvalCaseSet;
import com.ragagent.rag.RetrievedSegment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 生成侧评测：把正例跑一遍完整回答链路，度量"答案忠不忠于资料"。
 *
 * 为什么必须和 /eval 分开成两个接口：
 *   /eval        只测检索层，不调大模型 —— 秒级、可复现、每次改代码都能跑
 *   /eval-answer 要调大模型 —— 分钟级、有随机性、成本高，只能定期跑
 * 把两者混在一起，结果是快的那个被慢的拖死，可复现的那个变得不可复现。
 *
 * 这条链路上有三个不同的失败，必须分开度量，不能只看一个"准确率"：
 *   漏答   —— 正例被检索层拒答了（检索没召回），回答是"资料中没有相关内容"
 *   无引用 —— 答了，但没标引用编号，人工无法核对依据
 *   不忠实 —— 标了引用，但里面有资料支撑不了的断言（幻觉）
 */
@Service
public class AnswerEvaluationService {

    private static final Logger log = LoggerFactory.getLogger(AnswerEvaluationService.class);

    private final AnswerService answerService;
    private final FaithfulnessChecker faithfulnessChecker;

    public AnswerEvaluationService(AnswerService answerService, FaithfulnessChecker faithfulnessChecker) {
        this.answerService = answerService;
        this.faithfulnessChecker = faithfulnessChecker;
    }

    /**
     * @param limit 最多评测多少条正例（0 表示全部）。评测要调两次大模型，条数越多越慢。
     */
    public Map<String, Object> evaluate(int limit) {
        List<EvalCase> positives = new ArrayList<>(EvalCaseSet.CASES.stream()
                .filter(EvalCase::shouldAnswer)
                .toList());
        if (limit > 0 && limit < positives.size()) {
            positives = new ArrayList<>(positives.subList(0, limit));
        }

        int answered = 0;
        int abstained = 0;
        int withCitation = 0;
        int withInvalidCitation = 0;
        int judged = 0;
        int faithful = 0;
        int unjudged = 0;

        List<Map<String, Object>> details = new ArrayList<>(positives.size());

        for (EvalCase evalCase : positives) {
            AnswerService.AnswerResult result = answerService.answer(evalCase.query());

            Map<String, Object> detail = new LinkedHashMap<>();
            detail.put("query", evalCase.query());
            detail.put("expectedFile", evalCase.expectedFile());
            detail.put("answer", result.answer());
            detail.put("refused", result.refused());

            if (result.refused()) {
                // 正例被拒答 = 检索没把该文档捞出来，属于漏答，单独计数
                abstained++;
                detail.put("outcome", "abstained");
                details.add(detail);
                log.info("[eval-answer] 漏答 Q={}", evalCase.query());
                continue;
            }

            answered++;
            FaithfulnessChecker.CitationCheck citation =
                    faithfulnessChecker.checkCitations(result.answer(), result.hits().size());
            detail.put("citedIndexes", citation.citedIndexes());
            detail.put("invalidCitations", citation.invalidCitations());
            if (citation.hasValidCitation()) {
                withCitation++;
            }
            if (citation.invalidCitations() > 0) {
                withInvalidCitation++;
            }

            List<String> sourceTexts = result.hits().stream().map(RetrievedSegment::text).toList();
            FaithfulnessChecker.FaithfulnessVerdict verdict =
                    faithfulnessChecker.judge(evalCase.query(), sourceTexts, result.answer());
            detail.put("faithful", verdict.faithful());
            detail.put("reason", verdict.reason());
            detail.put("unsupportedClaims", verdict.unsupportedClaims());

            if (verdict.faithful() == null) {
                unjudged++;
                detail.put("outcome", "unjudged");
            } else {
                judged++;
                if (verdict.faithful()) {
                    faithful++;
                }
                detail.put("outcome", verdict.faithful() ? "faithful" : "unfaithful");
            }
            details.add(detail);

            log.info("[eval-answer] Q={} 引用={} 忠实={}", evalCase.query(), citation.citedIndexes(), verdict.faithful());
        }

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("evaluated", positives.size());
        result.put("answered", answered);
        result.put("abstained", abstained);
        result.put("withCitation", withCitation);
        result.put("withInvalidCitation", withInvalidCitation);
        result.put("judged", judged);
        result.put("faithful", faithful);
        result.put("unjudged", unjudged);

        result.put("citationCoverage", ratio(withCitation, answered));
        result.put("faithfulnessRate", ratio(faithful, judged));
        result.put("abstainRate", ratio(abstained, positives.size()));
        result.put("details", details);
        return result;
    }

    private static double ratio(int numerator, int denominator) {
        return denominator == 0 ? 0.0 : Math.round((double) numerator / denominator * 10000) / 10000.0;
    }
}
