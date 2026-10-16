package com.ragagent.agent;

import java.util.List;

/**
 * 一次 Agent 调用的完整结果。
 *
 * 为什么要把 toolCalls / rounds 一起返回，而不是只回最终答案：
 * 只看答案的话，你分不清它是"查了知识库"还是"自己编的"。这两者在用户眼里一模一样，
 * 在排障时却是天壤之别。把工具调用轨迹暴露出来，问题才可定位。
 *
 * @param answer           模型给出的最终回答
 * @param toolCalls        这一轮实际发生过的工具调用（按发生顺序）
 * @param rounds           实际用掉的对话轮次
 * @param maxRoundsReached 是否撞到了轮次上限（true 表示回答可能是被强制收尾的）
 */
public record AgentResult(String answer, List<ToolCall> toolCalls, int rounds, boolean maxRoundsReached) {

    /** 一次工具调用：模型要调什么、传了什么参数、工具回了什么。 */
    public record ToolCall(String name, String arguments, String result) {
    }
}
