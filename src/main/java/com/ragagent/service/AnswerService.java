package com.ragagent.service;

import com.ragagent.rag.RetrievedSegment;
import dev.langchain4j.model.chat.ChatModel;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * RAG 的读取侧：检索 → 拼 prompt → 生成回答。
 *
 * 为什么从 AskController 里抽出来：生成侧评测（/eval-answer）必须复用**完全同一条**回答链路，
 * 否则评的就不是线上跑的东西。逻辑留在 Controller 里，要么复制一份，要么没法评。
 *
 * prompt 里新增了引用编号的要求 —— 没有它，"答案是否忠于资料"这件事就只能靠模型自觉，
 * 既无法人工检查也无法自动度量。有了编号，规则层才有东西可校验。
 */
@Service
public class AnswerService {

    /** 检索不到够格资料时的固定话术。 */
    public static final String NO_ANSWER = "资料中没有相关内容";

    private static final String INSTRUCTION =
            "你是一个知识库问答助手。只能根据下面提供的【参考资料】回答用户问题。必须遵守：\n"
                    + "1. 如果参考资料里没有相关信息，直接回答'资料中没有相关内容'，严禁编造或凭常识补充；\n"
                    + "2. 回答中每一条结论都要用 [1][2] 这样的编号标注它依据的是第几段资料；\n"
                    + "3. 只允许引用存在的编号，不要引用不存在的编号。";

    private final RetrievalService retrievalService;
    private final ChatModel chatModel;

    public AnswerService(RetrievalService retrievalService, ChatModel chatModel) {
        this.retrievalService = retrievalService;
        this.chatModel = chatModel;
    }

    /**
     * 回答一个问题。
     *
     * @return 回答、实际喂给大模型的片段、以及是否走了拒答分支（拒答时不调用大模型）
     */
    public AnswerResult answer(String question) {
        List<RetrievedSegment> hits = retrievalService.retrieve(question);
        if (hits.isEmpty()) {
            return new AnswerResult(NO_ANSWER, List.of(), true);
        }
        return new AnswerResult(chatModel.chat(buildPrompt(hits, question)), hits, false);
    }

    private String buildPrompt(List<RetrievedSegment> hits, String question) {
        StringBuilder references = new StringBuilder();
        for (int i = 0; i < hits.size(); i++) {
            references.append('[').append(i + 1).append("] ").append(hits.get(i).text()).append('\n');
        }
        return INSTRUCTION
                + "\n\n【参考资料】\n" + references
                + "\n【问题】\n" + question;
    }

    /**
     * 一次回答的完整结果。
     *
     * @param answer  模型给出的回答（拒答时是固定话术）
     * @param hits    真正喂给模型的片段，顺序即 prompt 里 [1][2] 的编号顺序
     * @param refused 是否在检索层就拒答了
     */
    public record AnswerResult(String answer, List<RetrievedSegment> hits, boolean refused) {
    }
}
