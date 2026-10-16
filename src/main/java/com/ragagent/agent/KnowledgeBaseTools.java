package com.ragagent.agent;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.ragagent.rag.RetrievedSegment;
import com.ragagent.service.RetrievalService;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

/**
 * 暴露给大模型的工具集合。
 *
 * 三个工具是刻意配的：
 *   searchKnowledgeBase —— 真正干活的那个，把 RetrievalService 的整条链路（混合召回 + RRF + 精排 + 双闸门）
 *                          包装成一个工具。没有它这个项目只是个固定流程的 RAG；有了它，模型才需要在
 *                          "查还是不查""查什么"之间做选择。
 *   calculator          —— 让"模型会不会乱心算"变成可观测的：它必须调工具拿结果。
 *   currentTime         —— 同上，让"模型知不知道自己没有实时信息"变成可观测的。
 *
 * 后两个工具本身没有技术含量，但它们的价值在于**证明工具选择确实由模型做出**：
 * 用户问"现在几点"，如果模型知道该调 currentTime 而不是硬编一个时间，说明这条链路是活的。
 */
@Component
public class KnowledgeBaseTools {

    private static final DateTimeFormatter TIME_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private final RetrievalService retrievalService;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public KnowledgeBaseTools(RetrievalService retrievalService) {
        this.retrievalService = retrievalService;
    }

    /** 交给大模型的工具清单。顺序无所谓，模型按 description 自己挑。 */
    public List<AgentTool> all() {
        return List.of(searchKnowledgeBase(), calculator(), currentTime());
    }

    // ------------------------------------------------------------------

    private AgentTool searchKnowledgeBase() {
        ToolSpecification specification = ToolSpecification.builder()
                .name("searchKnowledgeBase")
                .description("在私有知识库中做语义检索，返回与问题最相关的文档原文片段。"
                        + "凡是涉及知识库内容、技术笔记、项目资料的问题，都必须先调用它，不要凭常识回答。")
                .parameters(JsonObjectSchema.builder()
                        .addStringProperty("query", "要在知识库里检索的问题，写成一句完整、可独立理解的话")
                        .required("query")
                        .build())
                .build();

        return new AgentTool(specification, request -> {
            String query = stringArgument(request, "query");
            List<RetrievedSegment> hits = retrievalService.retrieve(query);
            if (hits.isEmpty()) {
                return "知识库中没有检索到与该问题相关的内容。";
            }
            StringBuilder sb = new StringBuilder("知识库检索到 ").append(hits.size()).append(" 段资料：");
            for (int i = 0; i < hits.size(); i++) {
                RetrievedSegment segment = hits.get(i);
                sb.append("\n\n[").append(i + 1).append("] 来源=")
                        .append(segment.fileName()).append("#").append(segment.segmentIndex())
                        .append("\n").append(segment.text());
            }
            return sb.toString();
        });
    }

    private AgentTool calculator() {
        ToolSpecification specification = ToolSpecification.builder()
                .name("calculator")
                .description("计算一个四则运算表达式，支持 + - * / 与括号。"
                        + "凡是涉及算术，都必须用它拿结果，不要自己心算。")
                .parameters(JsonObjectSchema.builder()
                        .addStringProperty("expression", "只含数字与 + - * / ( ) 的表达式，例如 (12+34)*5/2")
                        .required("expression")
                        .build())
                .build();

        return new AgentTool(specification, request -> {
            String expression = stringArgument(request, "expression");
            double value = ArithmeticEvaluator.evaluate(expression);
            return expression + " = " + formatNumber(value);
        });
    }

    private AgentTool currentTime() {
        ToolSpecification specification = ToolSpecification.builder()
                .name("currentTime")
                .description("获取服务器当前的日期和时间。涉及“今天”“现在”“还有几天”这类问题时必须调用它。")
                .parameters(JsonObjectSchema.builder().build())
                .build();

        return new AgentTool(specification, request -> LocalDateTime.now().format(TIME_FORMAT));
    }

    // ------------------------------------------------------------------

    /**
     * 从模型给的 arguments（一个 JSON 字符串）里取出字符串参数。
     *
     * 模型给的参数是**不可信输入**：可能是 null、可能不是合法 JSON、可能少传字段。
     * 这里一律翻成 IllegalArgumentException，由 AgentService 转成文字回填，让模型自己纠错。
     */
    private String stringArgument(ToolExecutionRequest request, String key) {
        String arguments = request.arguments();
        if (arguments == null || arguments.isBlank()) {
            throw new IllegalArgumentException("没有传入任何参数，缺少 " + key);
        }
        JsonNode value;
        try {
            value = objectMapper.readTree(arguments).path(key);
        } catch (Exception e) {
            throw new IllegalArgumentException("参数不是合法 JSON：" + arguments, e);
        }
        if (value.isMissingNode() || value.isNull()) {
            throw new IllegalArgumentException("缺少参数 " + key);
        }
        return value.asText();
    }

    /** 整数不显示成 3.0，小数保留原样。 */
    private static String formatNumber(double value) {
        if (!Double.isInfinite(value) && !Double.isNaN(value)
                && value == Math.rint(value) && Math.abs(value) < 1e15) {
            return String.valueOf((long) value);
        }
        return String.valueOf(value);
    }
}
