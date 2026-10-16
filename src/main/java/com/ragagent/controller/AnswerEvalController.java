package com.ragagent.controller;

import com.ragagent.service.AnswerEvaluationService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

/**
 * 生成侧评测接口（要调用大模型，慢、有随机性，和 /eval 刻意分开）。
 *
 * 它回答的是"检索对了之后，答案到底忠不忠于资料"这个问题：
 *   漏答率     —— 正例被检索层拒答的比例（检索没召回）
 *   引用覆盖率 —— 回答里标了有效引用编号的比例
 *   忠实率     —— 判定为"每句都有资料依据"的比例（LLM-as-judge）
 *
 * 用法：GET /eval-answer          全量（42 条正例，约几分钟）
 *       GET /eval-answer?limit=10 只跑前 10 条，用于冒烟
 */
@RestController
public class AnswerEvalController {

    private final AnswerEvaluationService answerEvaluationService;

    public AnswerEvalController(AnswerEvaluationService answerEvaluationService) {
        this.answerEvaluationService = answerEvaluationService;
    }

    @GetMapping("/eval-answer")
    public Map<String, Object> evalAnswer(@RequestParam(value = "limit", defaultValue = "0") int limit) {
        return answerEvaluationService.evaluate(limit);
    }
}
