package com.ragagent;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

/**
 * 整个 Spring 容器的启动测试。
 *
 * 注意：这个测试必须能连上 pgvector（docker compose up -d），否则 embeddingStore bean
 * 会在建表阶段就失败。它被 @Disabled 是为了让 `mvn test` 不依赖外部环境 ——
 * 真正的业务逻辑由 RetrievalServiceTest / DocumentServiceTest 这两个纯单测覆盖。
 *
 * 要跑它：先 docker compose up -d，然后去掉 @Disabled（或改成 @Tag("integration")）。
 */
@SpringBootTest
@Disabled("需要本地 pgvector 容器（docker compose up -d）；默认不参与 mvn test")
class RagAgentApplicationTests {

    @Test
    void contextLoads() {
    }
}
