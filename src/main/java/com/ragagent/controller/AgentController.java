package com.ragagent.controller;

import com.ragagent.agent.AgentResult;
import com.ragagent.service.AgentService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Agent 入口：把"要不要调工具、调哪个"交给大模型自己决定。
 *
 * 它和 /ask 并存，不是替换关系：
 *   /ask    流程固定、结果可预测，/eval 的四个指标全部建立在它上面（换掉就会污染历史基线）；
 *   /agent  模型自由决策，更适合"问一句话，它自己去查/去算"的场景。
 * 两个接口放在一起，也方便直接对比"固定流程"和"模型决策"的差别。
 *
 * 用法：GET /agent?question=我的项目用了什么技术栈
 */
@RestController
public class AgentController {

    private final AgentService agentService;

    public AgentController(AgentService agentService) {
        this.agentService = agentService;
    }

    @GetMapping("/agent")
    public Map<String, Object> agent(@RequestParam("question") String question) {
        AgentResult result = agentService.ask(question);

        List<Map<String, Object>> toolCalls = new ArrayList<>();
        for (AgentResult.ToolCall call : result.toolCalls()) {
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("name", call.name());
            item.put("arguments", call.arguments());
            item.put("result", call.result());
            toolCalls.add(item);
        }

        Map<String, Object> response = new LinkedHashMap<>();
        response.put("answer", result.answer());
        response.put("toolCalls", toolCalls);
        response.put("rounds", result.rounds());
        response.put("maxRoundsReached", result.maxRoundsReached());
        return response;
    }
}
