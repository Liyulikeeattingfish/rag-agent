package com.ragagent.controller;

import com.ragagent.bench.VectorIndexBenchmark;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.sql.SQLException;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 向量索引压测接口。
 *
 * 用途：回答"你的索引是怎么设计的、ivf 还是 hnsw、召回掉了多少"这类问题 ——
 * 靠的是同一批数据上三种方案（精确扫描 / HNSW / IVFFlat）的实测对比，而不是背参数含义。
 *
 * 注意：它会 DROP 并重建 bench_vectors 这张压测专用表，不碰业务表 rag_segments。
 * rows 设了上限，避免一次请求把库灌爆（这是本地开发接口，不暴露到公网）。
 *
 * 用法：GET /bench/vector-index?rows=20000&queries=200&topK=10&clusters=100
 */
@RestController
@RequestMapping("/bench")
public class BenchController {

    /** 单次压测最多允许灌入的向量条数。 */
    private static final int MAX_ROWS = 200_000;

    private final VectorIndexBenchmark benchmark;

    public BenchController(VectorIndexBenchmark benchmark) {
        this.benchmark = benchmark;
    }

    @GetMapping("/vector-index")
    public Map<String, Object> vectorIndex(@RequestParam(value = "rows", defaultValue = "20000") int rows,
                                           @RequestParam(value = "queries", defaultValue = "200") int queries,
                                           @RequestParam(value = "topK", defaultValue = "10") int topK,
                                           @RequestParam(value = "clusters", defaultValue = "100") int clusters,
                                           @RequestParam(value = "noise", defaultValue = "0.3") double noise)
            throws SQLException {
        if (rows < 100 || rows > MAX_ROWS) {
            throw new IllegalArgumentException("rows 必须在 100 ~ " + MAX_ROWS + " 之间");
        }
        if (queries < 1 || queries > 2000) {
            throw new IllegalArgumentException("queries 必须在 1 ~ 2000 之间");
        }
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("request", Map.of("rows", rows, "queries", queries, "topK", topK, "clusters", clusters));
        response.put("benchmark", benchmark.run(rows, queries, topK, clusters, noise));
        return response;
    }
}
