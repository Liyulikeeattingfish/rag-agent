package com.ragagent.agent;

import com.ragagent.rag.RetrievedSegment;
import com.ragagent.service.RetrievalService;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.when;

/** 工具层：检索结果怎么交给模型、参数怎么解析、坏参数怎么报错。 */
@ExtendWith(MockitoExtension.class)
class KnowledgeBaseToolsTest {

    @Mock
    private RetrievalService retrievalService;

    private KnowledgeBaseTools tools;

    @BeforeEach
    void setUp() {
        tools = new KnowledgeBaseTools(retrievalService);
    }

    private AgentTool tool(String name) {
        return tools.all().stream().filter(item -> item.name().equals(name)).findFirst().orElseThrow();
    }

    private static ToolExecutionRequest request(String arguments) {
        return ToolExecutionRequest.builder().id("call-1").name("tool").arguments(arguments).build();
    }

    @Test
    @DisplayName("对外暴露三个工具，名字与模型看到的规格一致")
    void exposesThreeTools() {
        assertThat(tools.all()).extracting(AgentTool::name)
                .containsExactlyInAnyOrder("searchKnowledgeBase", "calculator", "currentTime");
    }

    @Test
    @DisplayName("检索命中：把原文连同来源一起交给模型，模型才有依据可引用")
    void searchReturnsSegmentsWithSource() throws Exception {
        when(retrievalService.retrieve(anyString())).thenReturn(List.of(
                new RetrievedSegment("RAG 是检索增强生成", "sample.txt", 0, 0.84, 0.05, 0.03, 0.6),
                new RetrievedSegment("pgvector 用来存向量", "db.md", 1, 0.77, null, 0.02, 0.5)));

        String result = tool("searchKnowledgeBase").handler()
                .execute(request("{\"query\":\"RAG 是什么\"}"));

        assertThat(result).contains("2 段资料").contains("sample.txt#0").contains("RAG 是检索增强生成")
                .contains("db.md#1").contains("pgvector 用来存向量");
    }

    @Test
    @DisplayName("检索为空：明确回一句‘没有相关内容’，而不是给空字符串（否则模型会以为是它没看懂）")
    void searchReturnsExplicitEmptyMessage() throws Exception {
        when(retrievalService.retrieve(anyString())).thenReturn(List.of());

        String result = tool("searchKnowledgeBase").handler().execute(request("{\"query\":\"今天天气\"}"));

        assertThat(result).contains("没有检索到");
    }

    @Test
    @DisplayName("calculator 走求值器；整数不带小数点")
    void calculatorEvaluatesExpression() throws Exception {
        AgentTool calculator = tool("calculator");

        assertThat(calculator.handler().execute(request("{\"expression\":\"(12+34)*5/2\"}")))
                .isEqualTo("(12+34)*5/2 = 115");
        assertThat(calculator.handler().execute(request("{\"expression\":\"1/2\"}")))
                .isEqualTo("1/2 = 0.5");
    }

    @Test
    @DisplayName("参数缺失、不是合法 JSON、为 null 都抛 IllegalArgumentException，交给上层回填")
    void rejectsMalformedArguments() {
        AgentTool calculator = tool("calculator");

        assertThatThrownBy(() -> calculator.handler().execute(request("{}")))
                .isInstanceOf(IllegalArgumentException.class).hasMessageContaining("expression");
        assertThatThrownBy(() -> calculator.handler().execute(request("not-a-json")))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> calculator.handler().execute(request(null)))
                .isInstanceOf(IllegalArgumentException.class);
    }
}
