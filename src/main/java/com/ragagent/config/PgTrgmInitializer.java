package com.ragagent.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

/**
 * 启动时准备 pg_trgm 扩展与倒排索引。
 *
 * 为什么做成启动任务而不是写在建表脚本里：pgvector 的表是 EmbeddingConfig 里
 * createTable(true) 自动建的，没有 SQL 迁移脚本可写。启动任务能保证一个顺序——
 * 表先建好（bean 实例化阶段），再建扩展和索引（Runner 阶段，在所有 bean 就绪之后）。
 *
 * 失败策略（刻意区分对待）：
 *   扩展建不出来 → 抛异常，启动失败。因为关键词通道没了扩展就是彻底不可用，
 *                 带着一个半死的检索链路跑起来，比启动失败更危险（问题会被藏起来）。
 *   索引建不出来 → 记 ERROR 但不拦截启动。原因见 KeywordSearcher 里的说明：
 *                 当前查询用的是 similarity() 而非 % 运算符，索引不在查询的关键路径上，
 *                 缺了它功能正确、只是变慢。但必须把错误喊出来，不能静默降级。
 */
@Component
public class PgTrgmInitializer implements ApplicationRunner {

    private static final Logger log = LoggerFactory.getLogger(PgTrgmInitializer.class);

    private static final String TABLE = "rag_segments";

    private final JdbcTemplate jdbcTemplate;

    public PgTrgmInitializer(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    @Override
    public void run(ApplicationArguments args) {
        try {
            jdbcTemplate.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm");
            log.info("[pg_trgm] 扩展已就绪");
        } catch (Exception e) {
            throw new IllegalStateException(
                    "pg_trgm 扩展创建失败，关键词检索通道无法工作。"
                            + "请确认数据库账号有权限执行 CREATE EXTENSION（pg_trgm 属于 trusted 扩展，"
                            + "PG13+ 里库 owner 即可创建）。原始错误：" + e.getMessage(), e);
        }

        try {
            jdbcTemplate.execute("CREATE INDEX IF NOT EXISTS idx_rag_segments_text_trgm "
                    + "ON " + TABLE + " USING gin (text gin_trgm_ops)");
            log.info("[pg_trgm] GIN 索引 idx_rag_segments_text_trgm 已就绪");
        } catch (Exception e) {
            log.error("[pg_trgm] GIN 索引创建失败（功能可用，关键词通道会退化为顺序扫描）：{}", e.getMessage());
        }
    }
}
