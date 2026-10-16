package com.ragagent.controller;

import com.ragagent.rag.RetrievedSegment;
import com.ragagent.service.AnswerService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 检索问答接口（RAG 的读取侧）。
 *
 * 本类现在只做两件事：调 AnswerService、把结果拼成 HTTP 响应。
 * 检索与生成都在 Service 里 —— 因为 /eval-answer 必须复用同一条回答链路，
 * 逻辑留在 Controller 里就会变成两份实现，评出来的就不是线上跑的东西。
 *
 * 用法：GET /ask?question=RAG是什么
 *
 * 返回体（结构向后兼容，只增不删）：
 *   answer      大模型回答
 *   retrieved   真正喂给大模型的原文数组
 *   sources     与 retrieved 一一对应的出处信息
 *   rerankUsed  本次是否真的做了精排（rerank 服务异常时会降级为 false）
 */
@RestController
public class AskController {

    private final AnswerService answerService;

    public AskController(AnswerService answerService) {
        this.answerService = answerService;
    }

    @GetMapping("/ask")
    public Map<String, Object> ask(@RequestParam("question") String question) {
        AnswerService.AnswerResult result = answerService.answer(question);

        List<String> retrieved = new ArrayList<>(result.hits().size());
        List<Map<String, Object>> sources = new ArrayList<>(result.hits().size());
        boolean rerankUsed = false;

        for (RetrievedSegment hit : result.hits()) {
            retrieved.add(hit.text());
            if (hit.rerankScore() != null) {
                rerankUsed = true;
            }

            Map<String, Object> source = new LinkedHashMap<>();
            source.put("filename", hit.fileName());
            source.put("segmentIndex", hit.segmentIndex());
            // score 是改造前就存在的旧字段，语义上等于向量分数，保留它以免破坏已有调用方。
            // 纯关键词命中的片段没走过向量通道，这里会是 null —— null 表示"没参与"，不是"算出来是 0"。
            source.put("score", round4(hit.vectorScore()));
            source.put("vectorScore", round4(hit.vectorScore()));
            source.put("keywordScore", round4(hit.keywordScore()));
            source.put("rrfScore", round4(hit.rrfScore()));
            source.put("rerankScore", round4(hit.rerankScore()));
            sources.add(source);
        }

        Map<String, Object> response = new LinkedHashMap<>();
        response.put("answer", result.answer());
        response.put("retrieved", retrieved);
        response.put("sources", sources);
        response.put("rerankUsed", rerankUsed);
        return response;
    }

    private static Double round4(Double value) {
        return value == null ? null : Math.round(value * 10000) / 10000.0;
    }
}
