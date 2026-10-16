package com.ragagent.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

/**
 * 启动时为向量列建 HNSW 索引。
 *
 * 为什么需要它：pgvector 不建索引时走的是精确扫描——召回 100%，但延迟随行数线性增长。
 * 索引把查询复杂度降下来，代价是召回不再是 100%，这个代价必须被测量（见 /bench/vector-index）。
 *
 * 参数选择：
 *   m = 16                  pgvector 默认值，图密度与索引体积的折中
 *   ef_construction = 64    默认值，构建质量与构建时间的折中
 *   ef_search 不在这里设   它是查询期参数（SET hnsw.ef_search），索引不动就能调
 *
 * 失败策略与 PgTrgmInitializer 一致：建不出来只记 ERROR，不拦截启动 ——
 * 没有索引检索依然正确，只是变慢；但错误必须喊出来，不能静默降级。
 */
@Component
public class VectorIndexInitializer implements ApplicationRunner {

    private static final Logger log = LoggerFactory.getLogger(VectorIndexInitializer.class);

    private static final String TABLE = "rag_segments";
    private static final String INDEX = "idx_rag_segments_embedding_hnsw";

    private final JdbcTemplate jdbcTemplate;

    public VectorIndexInitializer(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    @Override
    public void run(ApplicationArguments args) {
        try {
            jdbcTemplate.execute("CREATE INDEX IF NOT EXISTS " + INDEX
                    + " ON " + TABLE + " USING hnsw (embedding vector_cosine_ops)"
                    + " WITH (m = 16, ef_construction = 64)");
            log.info("[pgvector] HNSW 索引 {} 已就绪", INDEX);
        } catch (Exception e) {
            log.error("[pgvector] HNSW 索引创建失败（功能可用，向量检索会退化为精确扫描）：{}", e.getMessage());
        }
    }
}
