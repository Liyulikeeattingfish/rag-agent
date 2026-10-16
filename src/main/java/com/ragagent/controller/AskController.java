package com.ragagent.controller;

import com.ragagent.rag.RetrievedSegment;
import com.ragagent.service.RetrievalService;
import dev.langchain4j.model.chat.ChatModel;
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
 * 本类现在只做两件事：调检索、拼 prompt。所有检索的复杂度（混合召回 / RRF / 精排）
 * 都下沉到了 RetrievalService —— Controller 里写检索算法是上一版的事，
 * 现在检索逻辑要被 /ask 和 /eval 同时复用，再放在 Controller 里就会变成复制粘贴。
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

    /** 检索不到够格资料时的固定话术，与 prompt 里的要求保持一致（这行字本身不改判分逻辑）。 */
    private static final String NO_ANSWER = "资料中没有相关内容";

    private static final String INSTRUCTION =
            "你是一个知识库问答助手。只能根据下面提供的【参考资料】回答用户问题。"
                    + "如果参考资料里没有相关信息，必须直接回答'资料中没有相关内容'，严禁自己编造或凭常识补充。";

    private final RetrievalService retrievalService;
    private final ChatModel chatModel;

    public AskController(RetrievalService retrievalService, ChatModel chatModel) {
        this.retrievalService = retrievalService;
        this.chatModel = chatModel;
    }

    @GetMapping("/ask")
    public Map<String, Object> ask(@RequestParam("question") String question) {
        // 1. 混合召回 → RRF → 精排 → topN（全部在 RetrievalService 里，含各种开关与降级）
        List<RetrievedSegment> hits = retrievalService.retrieve(question);

        // 2. 一条都没过闸 → 直接拒答，不把不相关的资料塞给大模型
        if (hits.isEmpty()) {
            Map<String, Object> nothing = new LinkedHashMap<>();
            nothing.put("answer", NO_ANSWER);
            nothing.put("retrieved", List.of());
            nothing.put("sources", List.of());
            nothing.put("rerankUsed", false);
            return nothing;
        }

        List<String> retrieved = new ArrayList<>(hits.size());
        List<Map<String, Object>> sources = new ArrayList<>(hits.size());
        boolean rerankUsed = false;

        for (RetrievedSegment hit : hits) {
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

        // 3. 只把最终候选拼进 prompt
        String answer = chatModel.chat(buildPrompt(retrieved, question));

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("answer", answer);
        result.put("retrieved", retrieved);
        result.put("sources", sources);
        result.put("rerankUsed", rerankUsed);
        return result;
    }

    private String buildPrompt(List<String> retrieved, String question) {
        StringBuilder refs = new StringBuilder();
        for (int i = 0; i < retrieved.size(); i++) {
            refs.append(i + 1).append(". ").append(retrieved.get(i)).append("\n");
        }
        return INSTRUCTION
                + "\n\n【参考资料】\n" + refs
                + "\n【问题】\n" + question;
    }

    private static Double round4(Double value) {
        return value == null ? null : Math.round(value * 10000) / 10000.0;
    }
}
