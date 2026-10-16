package com.ragagent.service;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.filter.Filter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.jdbc.core.JdbcTemplate;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * 入库幂等规则的单元测试：内容级去重、同名冲突、显式覆盖。
 *
 * 这三条是纯业务规则，也是最容易在「加个字段/换个判断顺序」时被改坏的地方 ——
 * 一旦坏了不会报错，只会表现为「检索答案里混进旧版本的内容」，非常难排查。
 */
@ExtendWith(MockitoExtension.class)
class DocumentServiceTest {

    @Mock
    private EmbeddingModel embeddingModel;
    @Mock
    private EmbeddingStore<TextSegment> embeddingStore;
    @Mock
    private JdbcTemplate jdbcTemplate;

    private DocumentService service;

    @BeforeEach
    void setUp() {
        service = new DocumentService(embeddingModel, embeddingStore, jdbcTemplate);
    }

    /** 两个 COUNT 查询走的是同一个方法，只能按 SQL 内容区分谁是谁。 */
    private void stubCounts(int contentHashHits, int sameNameSegments) {
        when(jdbcTemplate.queryForObject(anyString(), eq(Integer.class), any()))
                .thenAnswer(invocation -> invocation.<String>getArgument(0).contains("content_hash")
                        ? contentHashHits
                        : sameNameSegments);
    }

    private void stubEmbeddingAndStore(int segments) {
        List<Embedding> embeddings = java.util.stream.IntStream.range(0, segments)
                .mapToObj(i -> Embedding.from(new float[]{0.1f, 0.2f}))
                .toList();
        when(embeddingModel.embedAll(anyList())).thenReturn(Response.from(embeddings));
        when(embeddingStore.addAll(anyList(), anyList())).thenReturn(
                java.util.stream.IntStream.range(0, segments).mapToObj(i -> "id-" + i).toList());
    }

    @Test
    @DisplayName("内容级去重：库里已有同样内容就直接跳过，与文件名无关")
    void duplicateContentIsSkipped() {
        stubCounts(1, 0);

        Map<String, Object> result = service.ingest("换个名字.txt", "已经入库过的内容", false);

        assertThat(result.get("status")).isEqualTo("duplicate");
        assertThat(result.get("success")).isEqualTo(false);
        verify(embeddingStore, never()).addAll(anyList(), anyList());
        verify(embeddingStore, never()).removeAll(any(Filter.class));
    }

    @Test
    @DisplayName("同名不同内容：默认判为冲突并拒绝入库，不覆盖已有版本")
    void sameNameDifferentContentIsRejectedByDefault() {
        stubCounts(0, 2);

        Map<String, Object> result = service.ingest("sample.txt", "改过的新正文", false);

        assertThat(result.get("status")).isEqualTo("conflict");
        assertThat(result.get("success")).isEqualTo(false);
        assertThat(result.get("existingSegments")).isEqualTo(2);
        verify(embeddingStore, never()).addAll(anyList(), anyList());
        verify(embeddingStore, never()).removeAll(any(Filter.class));
    }

    @Test
    @DisplayName("overwrite=true：先删干净旧版本再写新版本，避免新旧片段并存")
    void overwriteRemovesOldSegmentsBeforeWriting() {
        stubCounts(0, 2);
        stubEmbeddingAndStore(1);

        Map<String, Object> result = service.ingest("sample.txt", "改过的新正文", true);

        assertThat(result.get("status")).isEqualTo("updated");
        assertThat(result.get("removed")).isEqualTo(2);
        assertThat(result.get("segments")).isEqualTo(1);
        verify(embeddingStore).removeAll(any(Filter.class));
        verify(embeddingStore).addAll(anyList(), anyList());
    }

    @Test
    @DisplayName("全新文档：直接入库，状态为 created")
    void brandNewDocumentIsCreated() {
        stubCounts(0, 0);
        stubEmbeddingAndStore(1);

        Map<String, Object> result = service.ingest("new.txt", "全新的正文", false);

        assertThat(result.get("status")).isEqualTo("created");
        assertThat(result.get("success")).isEqualTo(true);
        assertThat(result).doesNotContainKey("removed");
        verify(embeddingStore, never()).removeAll(any(Filter.class));
        verify(embeddingStore).addAll(anyList(), anyList());
    }
}
