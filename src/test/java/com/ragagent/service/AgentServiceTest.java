package com.ragagent.service;

import com.ragagent.agent.AgentResult;
import com.ragagent.agent.AgentTool;
import com.ragagent.agent.KnowledgeBaseTools;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Agent 主循环的单元测试。只 mock 大模型和工具集，不碰网络与数据库。
 *
 * 这里钉的都是"错了不会报错、只会悄悄失灵"的地方：消息回填顺序、轮次上限、
 * 工具异常兜底、未知工具兜底 —— 它们在真机上表现为"模型变笨了"，很难查。
 */
@ExtendWith(MockitoExtension.class)
class AgentServiceTest {

    @Mock
    private ChatModel chatModel;
    @Mock
    private KnowledgeBaseTools tools;

    private AgentService service;

    @BeforeEach
    void setUp() {
        service = new AgentService(chatModel, tools, 5);
    }

    private static AgentTool tool(String name, AgentTool.ToolHandler handler) {
        ToolSpecification specification = ToolSpecification.builder()
                .name(name)
                .description(name + " 的说明")
                .parameters(JsonObjectSchema.builder().addStringProperty("query").required("query").build())
                .build();
        return new AgentTool(specification, handler);
    }

    private static ToolExecutionRequest call(String name) {
        return ToolExecutionRequest.builder().id("call-1").name(name).arguments("{\"query\":\"x\"}").build();
    }

    private static ChatResponse asksForTool(ToolExecutionRequest... requests) {
        return ChatResponse.builder().aiMessage(AiMessage.from(List.of(requests))).build();
    }

    private static ChatResponse answers(String text) {
        return ChatResponse.builder().aiMessage(AiMessage.from(text)).build();
    }

    @Test
    @DisplayName("模型不需要工具时直接作答，只走一轮")
    void answersWithoutToolCall() {
        when(tools.all()).thenReturn(List.of(tool("searchKnowledgeBase", request -> "hit")));
        when(chatModel.chat(any(ChatRequest.class))).thenReturn(answers("直接回答"));

        AgentResult result = service.ask("你好");

        assertThat(result.answer()).isEqualTo("直接回答");
        assertThat(result.rounds()).isEqualTo(1);
        assertThat(result.toolCalls()).isEmpty();
        assertThat(result.maxRoundsReached()).isFalse();
    }

    @Test
    @DisplayName("模型要求调工具：执行 → 结果回填 → 再问一轮 → 拿到最终答案")
    void executesToolAndFeedsResultBack() {
        when(tools.all()).thenReturn(List.of(tool("searchKnowledgeBase", request -> "知识库原文")));

        // 注意：ChatRequest 里存的是 messages 列表的**引用**，而循环还在继续往里追加。
        // 所以必须在每次调用的当下就快照一份，否则断言时看到的是最终状态、不是那一轮的输入。
        List<List<ChatMessage>> sentMessages = new ArrayList<>();
        when(chatModel.chat(any(ChatRequest.class))).thenAnswer(invocation -> {
            ChatRequest request = invocation.getArgument(0);
            sentMessages.add(new ArrayList<>(request.messages()));
            return sentMessages.size() == 1
                    ? asksForTool(call("searchKnowledgeBase"))
                    : answers("基于资料的回答");
        });

        AgentResult result = service.ask("RAG 是什么");

        assertThat(result.answer()).isEqualTo("基于资料的回答");
        assertThat(result.rounds()).isEqualTo(2);
        assertThat(result.toolCalls()).hasSize(1);
        assertThat(result.toolCalls().get(0).name()).isEqualTo("searchKnowledgeBase");
        assertThat(result.toolCalls().get(0).result()).isEqualTo("知识库原文");

        // 第二轮发给模型的消息必须是：system → user → assistant(要求调工具) → tool(结果)。
        // 少了 assistant 那条，OpenAI 兼容接口会直接报错 —— 这是手写循环最容易漏的一步。
        assertThat(sentMessages).hasSize(2);
        List<ChatMessage> secondRound = sentMessages.get(1);
        assertThat(secondRound).hasSize(4);
        assertThat(secondRound.get(2)).isInstanceOf(AiMessage.class);
        assertThat(((AiMessage) secondRound.get(2)).hasToolExecutionRequests()).isTrue();
        assertThat(secondRound.get(3)).isInstanceOf(ToolExecutionResultMessage.class);
    }

    @Test
    @DisplayName("工具抛异常不冒泡：转成文字回填，模型仍能给出回答")
    void toolFailureIsFedBackInsteadOfThrown() {
        when(tools.all()).thenReturn(List.of(tool("searchKnowledgeBase", request -> {
            throw new IllegalStateException("知识库连接失败");
        })));
        when(chatModel.chat(any(ChatRequest.class)))
                .thenReturn(asksForTool(call("searchKnowledgeBase")))
                .thenReturn(answers("检索暂时不可用"));

        AgentResult result = service.ask("RAG 是什么");

        assertThat(result.answer()).isEqualTo("检索暂时不可用");
        assertThat(result.toolCalls().get(0).result()).contains("工具执行失败").contains("知识库连接失败");
    }

    @Test
    @DisplayName("模型喊了不存在的工具：回填可用工具清单，不抛异常")
    void unknownToolIsReportedBack() {
        when(tools.all()).thenReturn(List.of(tool("searchKnowledgeBase", request -> "hit")));
        when(chatModel.chat(any(ChatRequest.class)))
                .thenReturn(asksForTool(call("deleteEverything")))
                .thenReturn(answers("换了个方式回答"));

        AgentResult result = service.ask("随便问问");

        assertThat(result.answer()).isEqualTo("换了个方式回答");
        assertThat(result.toolCalls().get(0).result()).contains("未知工具").contains("searchKnowledgeBase");
    }

    @Test
    @DisplayName("模型反复调工具：到上限即停，并摘掉工具再问一次收尾，防止死循环烧 token")
    void stopsAtMaxRoundsAndForcesFinalAnswer() {
        when(tools.all()).thenReturn(List.of(tool("searchKnowledgeBase", request -> "hit")));
        when(chatModel.chat(any(ChatRequest.class)))
                .thenReturn(asksForTool(call("searchKnowledgeBase")))
                .thenReturn(asksForTool(call("searchKnowledgeBase")))
                .thenReturn(asksForTool(call("searchKnowledgeBase")))
                .thenReturn(asksForTool(call("searchKnowledgeBase")))
                .thenReturn(asksForTool(call("searchKnowledgeBase")))
                .thenReturn(answers("强制收尾的回答"));

        AgentResult result = service.ask("没完没了的问题");

        assertThat(result.maxRoundsReached()).isTrue();
        assertThat(result.rounds()).isEqualTo(5);
        assertThat(result.answer()).isEqualTo("强制收尾的回答");
        assertThat(result.toolCalls()).hasSize(5);

        ArgumentCaptor<ChatRequest> captor = ArgumentCaptor.forClass(ChatRequest.class);
        verify(chatModel, times(6)).chat(captor.capture());
        // 最后一次调用不能带工具：否则模型又能继续要工具，轮次上限就形同虚设
        assertThat(captor.getAllValues().get(5).toolSpecifications()).isNullOrEmpty();
    }
}
