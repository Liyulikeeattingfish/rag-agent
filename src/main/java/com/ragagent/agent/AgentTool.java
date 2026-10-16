package com.ragagent.agent;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;

/**
 * 一个可被大模型调用的工具 = 「给模型看的说明书」+「真正干活的代码」。
 *
 * 为什么把这两样绑在一个对象里：它们天然必须一致 —— 模型只会照 specification 里的名字喊工具，
 * 而代码要按同一个名字找实现。拆成两张表维护，早晚会出现"改了名字忘了同步"，
 * 那种 bug 在运行时表现为"模型反复调一个不存在的工具"，从日志里很难一眼看出来。
 */
public record AgentTool(ToolSpecification specification, ToolHandler handler) {

    /**
     * 工具的实现。
     * 允许抛异常 —— 由 AgentService 统一兜住并转成一段文字回填给模型，
     * 而不是让异常打断整轮对话（外部调用抖动不该让 /agent 直接 500）。
     */
    @FunctionalInterface
    public interface ToolHandler {
        String execute(ToolExecutionRequest request) throws Exception;
    }

    public String name() {
        return specification.name();
    }
}
