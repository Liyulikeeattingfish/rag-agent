package com.ragagent.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.ragagent.config.RetrievalProperties;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

import java.time.Duration;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 硅基流动 rerank（精排）客户端。
 *
 * 为什么粗召回之后还要精排：粗召回用的是"双塔"结构——问题和文档各自独立编码成向量，
 * 彼此在计算前从未见过面，所以只能捕捉粗粒度的语义相似。cross-encoder 类模型
 * （bge-reranker）是把 query 和 document 拼在一起送进模型，能逐字对比，精度高得多，
 * 代价是每条候选都要跑一次推理，没法对全库做，只能对粗召回的那 10 条做。
 * 这就是业界通用的「粗召回 → 精排」两段式：用便宜的模型把范围从百万缩到十个，
 * 再用贵的模型把这十个排准。
 *
 * 接口：POST {siliconflow.base-url}/rerank
 * 请求：{ model, query, documents[], top_n, return_documents }
 * 响应：{ results: [ { index, relevance_score } ] }
 * index 是 documents 数组里的下标 —— 靠它把分数映射回原候选，千万别靠文档内容去匹配
 * （候选之间内容可能高度相似，按内容找会张冠李戴）。
 */
@Service
public class RerankClient {

    private final RestClient restClient;
    private final ObjectMapper objectMapper = new ObjectMapper();
    private final String rerankUrl;
    private final String model;

    public RerankClient(@Value("${siliconflow.base-url}") String baseUrl,
                        @Value("${siliconflow.api-key}") String apiKey,
                        RetrievalProperties properties) {
        // 超时一定要设：rerank 是外部 HTTP 调用，不设超时的话对端卡住会拖死整个 /ask 线程。
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(Duration.ofSeconds(10));
        factory.setReadTimeout(Duration.ofSeconds(60));

        // 不设 baseUrl 而是拼完整 URL，避免 RestClient 的 baseUrl 与以 / 开头的 path 拼出意外结果
        String normalized = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.rerankUrl = normalized + "/rerank";
        this.model = properties.getRerankModel();

        // 密钥只在这里从配置读一次，绝不写死在代码里；也不打印到日志。
        this.restClient = RestClient.builder()
                .requestFactory(factory)
                .defaultHeader("Authorization", "Bearer " + apiKey)
                .defaultHeader("Content-Type", "application/json")
                .build();
    }

    /**
     * 对候选文档精排。
     *
     * @return 下标 -> 相关性分数（0~1，越高越相关）。下标对应传入 documents 的顺序。
     */
    public Map<Integer, Double> rerank(String query, List<String> documents) {
        if (documents.isEmpty()) {
            return Map.of();
        }

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", model);
        body.put("query", query);
        body.put("documents", documents);
        // 我们本地已经有原文了，让服务端别再回传一份，省带宽
        body.put("return_documents", false);
        body.put("top_n", documents.size());

        String response = restClient.post()
                .uri(rerankUrl)
                .body(body)
                .retrieve()
                .body(String.class);

        Map<Integer, Double> scores = new HashMap<>();
        try {
            JsonNode root = objectMapper.readTree(response);
            for (JsonNode item : root.path("results")) {
                scores.put(item.path("index").asInt(), item.path("relevance_score").asDouble());
            }
        } catch (Exception e) {
            // 解析失败当成调用失败抛出去，由上层统一降级，别在这里吞掉
            throw new IllegalStateException("rerank 响应解析失败: " + abbreviate(response), e);
        }
        return scores;
    }

    private static String abbreviate(String text) {
        if (text == null) {
            return "null";
        }
        return text.length() <= 200 ? text : text.substring(0, 200) + "...";
    }
}
