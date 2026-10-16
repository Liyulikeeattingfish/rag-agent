package com.ragagent.service;

import com.ragagent.agent.AgentResult;
import com.ragagent.agent.AgentTool;
import com.ragagent.agent.KnowledgeBaseTools;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Agent 主循环：让大模型自己决定要不要调工具、调哪个、调几次。
 *
 * 和 /ask 的本质区别：
 *   /ask    是固定流程 —— 检索一定会发生，模型没有选择权，它只负责把检索结果写成人话。
 *   /agent  是模型自己决策 —— 它可能检索、可能算数、可能什么都不查直接回答。
 * 这也是"RAG"和"Agent"的分界线：**有没有由模型做出的选择**。
 *
 * 循环本身完全手写，没有用 AiServices 那套注解魔法，原因有二：
 *   1. 可控：轮次上限、异常兜底、未知工具兜底这些必须由我决定，不能藏在框架里；
 *   2. 可观测：每一轮调了什么、参数是什么、结果多长，都要能记下来 —— 否则线上出问题只能猜。
 *
 * 一轮的标准协议（OpenAI 兼容接口的 function calling）：
 *   user → assistant(要求调用工具) → tool(结果) → assistant(最终回答)
 * 注意 assistant 那条**必须回填进消息历史**，不回填的话下一轮模型看不到自己要求过什么，
 * 接口会直接报错。这是手写循环最容易踩的坑。
 */
@Service
public class AgentService {

    private static final Logger log = LoggerFactory.getLogger(AgentService.class);

    /**
     * 系统指令。必须把"什么情况下一定要调工具"写死 ——
     * 否则模型很容易凭常识直接回答，工具就形同虚设（尤其小模型）。
     */
    private static final String SYSTEM_PROMPT = """
            你是一个私有知识库助手，可以调用工具来完成任务。必须遵守以下规则：
            1. 凡是涉及知识库内容、技术笔记、项目资料的问题，必须先调用 searchKnowledgeBase 检索，禁止凭常识或记忆直接回答；
            2. 如果检索结果里没有能回答问题的内容，必须直接回答“资料中没有相关内容”，不要编造；
            3. 凡是涉及算术计算，必须调用 calculator 拿结果，不要自己心算；
            4. 凡涉及“今天”“现在”这类时间问题，必须调用 currentTime；
            5. 回答要简洁，并指出依据来自哪一段资料（用 [1][2] 这样的编号）。
            """;

    /** 兜底话术：只有连"摘掉工具再问一次"都拿不到文本时才会用到。 */
    static final String MAX_ROUNDS_ANSWER = "（已达到工具调用轮次上限，无法给出完整回答）";

    private final ChatModel chatModel;
    private final KnowledgeBaseTools tools;
    private final int maxRounds;

    public AgentService(ChatModel chatModel,
                        KnowledgeBaseTools tools,
                        @Value("${agent.max-rounds:5}") int maxRounds) {
        this.chatModel = chatModel;
        this.tools = tools;
        this.maxRounds = maxRounds;
    }

    public AgentResult ask(String question) {
        List<AgentTool> availableTools = tools.all();
        Map<String, AgentTool> toolIndex = new LinkedHashMap<>();
        for (AgentTool tool : availableTools) {
            toolIndex.put(tool.name(), tool);
        }
        List<ToolSpecification> specifications = availableTools.stream().map(AgentTool::specification).toList();

        List<ChatMessage> messages = new ArrayList<>();
        messages.add(SystemMessage.from(SYSTEM_PROMPT));
        messages.add(UserMessage.from(question));

        List<AgentResult.ToolCall> toolCalls = new ArrayList<>();

        for (int round = 1; round <= maxRounds; round++) {
            ChatResponse response = chatModel.chat(ChatRequest.builder()
                    .messages(messages)
                    .toolSpecifications(specifications)
                    .build());
            AiMessage aiMessage = response.aiMessage();

            // 关键一步：模型自己发出的这条消息也要放回历史（见类注释里的协议说明）
            messages.add(aiMessage);

            if (!aiMessage.hasToolExecutionRequests()) {
                log.info("[agent] Q={} 第 {} 轮直接作答，未调用工具", question, round);
                return new AgentResult(aiMessage.text(), toolCalls, round, false);
            }

            for (ToolExecutionRequest request : aiMessage.toolExecutionRequests()) {
                String result = execute(toolIndex.get(request.name()), request, toolIndex.keySet());
                messages.add(ToolExecutionResultMessage.from(request, result));
                toolCalls.add(new AgentResult.ToolCall(request.name(), request.arguments(), result));
                log.info("[agent] Q={} 第 {} 轮调用工具 {}，参数={}，返回 {} 字",
                        question, round, request.name(), request.arguments(), result.length());
            }
        }

        // 轮次用尽：把工具摘掉再问一次，逼模型用已经拿到的信息收尾。
        // 直接回一句"达到上限"对用户毫无价值 —— 它此时手里其实已经有资料了。
        log.warn("[agent] Q={} 达到工具调用轮次上限 {}，摘掉工具强制收尾", question, maxRounds);
        ChatResponse forced = chatModel.chat(ChatRequest.builder().messages(messages).build());
        String text = forced.aiMessage().text();
        return new AgentResult(
                text == null || text.isBlank() ? MAX_ROUNDS_ANSWER : text,
                toolCalls, maxRounds, true);
    }

    /**
     * 执行一次工具调用。**任何异常都在这里被转成文字**，绝不向上抛：
     * 工具失败应该让模型自己换个思路重试，而不是把整个 /agent 请求打断成 500。
     */
    private String execute(AgentTool tool, ToolExecutionRequest request, Set<String> availableNames) {
        if (tool == null) {
            // 模型喊了一个不存在的工具。同样回填一段说明，它下一轮通常会自己改正。
            return "未知工具 " + request.name() + "。当前可用工具只有：" + availableNames;
        }
        try {
            return tool.handler().execute(request);
        } catch (Exception e) {
            log.error("[agent] 工具 {} 执行失败：{}", request.name(), e.toString());
            return "工具执行失败：" + e.getMessage() + "。请换一种方式，或者直接说明无法完成。";
        }
    }
}
