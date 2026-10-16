package com.ragagent.config;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.pgvector.PgVectorEmbeddingStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import javax.sql.DataSource;
import java.time.Duration;

/**
 * 向量化配置类（RAG 的写入侧）。
 * 作用：
 *   1. EmbeddingModel：把文本转成向量，这里用硅基流动的 OpenAI 兼容接口。
 *   2. EmbeddingStore：把向量存进 PostgreSQL + pgvector。
 * 之后任何地方想做 embedding / 向量入库，直接注入这两个 bean 即可。
 */
@Configuration
public class EmbeddingConfig {

    @Value("${siliconflow.api-key}")
    private String apiKey;

    @Value("${siliconflow.base-url}")
    private String baseUrl;

    @Value("${siliconflow.model}")
    private String model;

    @Value("${siliconflow.dimension}")
    private Integer dimension;

    /**
     * 硅基流动 embedding 模型（OpenAI 兼容协议）。
     * 注意：base-url 必须带 /v1。
     */
    @Bean
    public EmbeddingModel embeddingModel() {
        return OpenAiEmbeddingModel.builder()
                .baseUrl(baseUrl)
                .apiKey(apiKey)
                .modelName(model)
                .dimensions(dimension)
                .timeout(Duration.ofSeconds(60))
                .build();
    }

    /**
     * pgvector 向量库。
     * 表名 rag_segments，维度 1024（必须与 embedding 输出维度一致）。
     * createTable(true)：首次启动自动建表（表已存在则跳过）。
     *
     * 说明：LangChain4j 1.0.0 GA 配套的 pgvector 模块（1.0.0-beta5）里，
     * 类名是 PgVectorEmbeddingStore，用 datasourceBuilder() 传入 DataSource，
     * 对应旧文档里的 PgVectorStore.builder(...) / initializeSchema(true)。
     */
    @Bean
    public EmbeddingStore<TextSegment> embeddingStore(DataSource dataSource) {
        return PgVectorEmbeddingStore.datasourceBuilder()
                .datasource(dataSource)
                .table("rag_segments")
                .dimension(dimension)
                .createTable(true)
                .build();
    }
}
