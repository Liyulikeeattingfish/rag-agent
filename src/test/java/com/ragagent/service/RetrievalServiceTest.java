package com.ragagent.service;

import com.ragagent.config.RetrievalProperties;
import com.ragagent.rag.KeywordSearcher;
import com.ragagent.rag.RagMetadataKeys;
import com.ragagent.rag.RetrievedSegment;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * 检索编排的单元测试。
 *
 * 只测「编排逻辑」本身（融合 / 排序 / 过滤 / 降级），不测向量库与外部服务 ——
 * 那两类依赖在这里都是 mock。这样单测不依赖 Docker 与网络，几秒跑完，
 * 能覆盖住改 RRF、改阈值、改降级策略时最容易踩坏的那几条分支。
 */
@ExtendWith(MockitoExtension.class)
class RetrievalServiceTest {

    @Mock
    private EmbeddingModel embeddingModel;
    @Mock
    private EmbeddingStore<TextSegment> embeddingStore;
    @Mock
    private KeywordSearcher keywordSearcher;
    @Mock
    private RerankClient rerankClient;

    private RetrievalProperties properties;
    private RetrievalService service;

    @BeforeEach
    void setUp() {
        properties = new RetrievalProperties();
        service = new RetrievalService(embeddingModel, embeddingStore, keywordSearcher, rerankClient, properties);
        when(embeddingModel.embed(anyString())).thenReturn(Response.from(Embedding.from(new float[]{0.1f, 0.2f})));
    }

    // ------------------------------------------------------------------
    // 造数据的小工具
    // ------------------------------------------------------------------

    private static EmbeddingMatch<TextSegment> match(String fileName, int index, double score, String text) {
        Metadata metadata = new Metadata()
                .put(RagMetadataKeys.FILE_NAME, fileName)
                .put(RagMetadataKeys.SEGMENT_INDEX, index);
        return new EmbeddingMatch<>(score, fileName + "#" + index,
                Embedding.from(new float[]{0.1f, 0.2f}), TextSegment.from(text, metadata));
    }

    private void stubVector(EmbeddingMatch<TextSegment>... matches) {
        when(embeddingStore.search(any(EmbeddingSearchRequest.class)))
                .thenReturn(new EmbeddingSearchResult<>(List.of(matches)));
    }

    private void stubKeyword(KeywordSearcher.KeywordHit... hits) {
        when(keywordSearcher.search(anyString(), anyInt(), anyDouble())).thenReturn(List.of(hits));
    }

    // ------------------------------------------------------------------
    // 粗召回：RRF 融合与去重
    // ------------------------------------------------------------------

    @Test
    @DisplayName("RRF 按名次融合：两个通道都命中的片段得分累加，排在只被单通道命中的前面")
    void rrfFusionAccumulatesScoresAcrossChannels() {
        properties.setRerankEnabled(false);
        // 向量通道：x 第 1、y 第 2
        stubVector(match("x.txt", 0, 0.80, "x 的正文"), match("y.txt", 0, 0.75, "y 的正文"));
        // 关键词通道：y 第 1、z 第 2
        stubKeyword(new KeywordSearcher.KeywordHit("y 的正文", "y.txt", 0, 0.09),
                new KeywordSearcher.KeywordHit("z 的正文", "z.txt", 0, 0.02));

        List<RetrievedSegment> result = service.retrieve("问题");

        assertThat(result).extracting(RetrievedSegment::fileName).containsExactly("y.txt", "x.txt", "z.txt");
        // y 被两个通道同时召回，两路分数都在，RRF 分最高
        assertThat(result.get(0).vectorScore()).isNotNull();
        assertThat(result.get(0).keywordScore()).isNotNull();
        assertThat(result.get(0).rrfScore()).isGreaterThan(result.get(1).rrfScore());
        // x 只走向量通道，z 只走关键词通道 —— 缺的那一路是 null，不是 0
        assertThat(result.get(1).keywordScore()).isNull();
        assertThat(result.get(2).vectorScore()).isNull();
    }

    @Test
    @DisplayName("去重身份是 文件名+片段序号，两段内容完全相同的片段不会被误合并")
    void dedupIdentityIsFileNameAndIndexNotText() {
        properties.setRerankEnabled(false);
        stubVector(match("a.txt", 0, 0.80, "两段一模一样的正文"),
                match("b.txt", 0, 0.75, "两段一模一样的正文"));

        List<RetrievedSegment> result = service.retrieve("问题");

        assertThat(result).hasSize(2);
        assertThat(result).extracting(RetrievedSegment::fileName).containsExactly("a.txt", "b.txt");
    }

    // ------------------------------------------------------------------
    // 精排：排序、阈值过滤、降级
    // ------------------------------------------------------------------

    @Test
    @DisplayName("精排成功：按 rerank 分重排，低于阈值的候选被丢弃")
    void rerankReordersAndFiltersByThreshold() {
        properties.setHybridEnabled(false);
        properties.setRerankEnabled(true);
        properties.setRerankMinScore(0.05);
        stubVector(match("a.txt", 0, 0.90, "a 的正文"), match("b.txt", 0, 0.80, "b 的正文"));
        when(rerankClient.rerank(anyString(), anyList())).thenReturn(Map.of(0, 0.9001, 1, 0.01));

        List<RetrievedSegment> result = service.retrieve("问题");

        assertThat(result).hasSize(1);
        assertThat(result.get(0).fileName()).isEqualTo("a.txt");
        assertThat(result.get(0).rerankScore()).isEqualTo(0.9001);
    }

    @Test
    @DisplayName("精排后候选全部低于阈值：返回空列表，交给调用方拒答")
    void allCandidatesBelowRerankThresholdMeansRefusal() {
        properties.setHybridEnabled(false);
        properties.setRerankEnabled(true);
        properties.setRerankMinScore(0.05);
        stubVector(match("a.txt", 0, 0.90, "a 的正文"), match("b.txt", 0, 0.80, "b 的正文"));
        when(rerankClient.rerank(anyString(), anyList())).thenReturn(Map.of(0, 0.001, 1, 0.0002));

        assertThat(service.retrieve("今天天气怎么样")).isEmpty();
    }

    @Test
    @DisplayName("精排服务异常：降级为 RRF 顺序，不抛异常，且不拿 null 分数去过滤（候选不被清空）")
    void rerankFailureFallsBackToRrfOrderWithoutFiltering() {
        properties.setHybridEnabled(false);
        properties.setRerankEnabled(true);
        properties.setRerankMinScore(0.05); // 阈值很高，若误用 null 比较会把候选全丢掉
        stubVector(match("a.txt", 0, 0.90, "a 的正文"), match("b.txt", 0, 0.80, "b 的正文"));
        when(rerankClient.rerank(anyString(), anyList())).thenThrow(new IllegalStateException("rerank 服务不可用"));

        List<RetrievedSegment> result = service.retrieve("问题");

        assertThat(result).hasSize(2);
        assertThat(result).extracting(RetrievedSegment::fileName).containsExactly("a.txt", "b.txt");
        assertThat(result).allSatisfy(segment -> assertThat(segment.rerankScore()).isNull());
    }

    // ------------------------------------------------------------------
    // 基线路径与空候选
    // ------------------------------------------------------------------

    @Test
    @DisplayName("两个开关都关 = 纯向量基线：只用向量通道，且 maxResults 取 finalTopK")
    void bothSwitchesOffFallsBackToVectorOnlyBaseline() {
        properties.setHybridEnabled(false);
        properties.setRerankEnabled(false);
        stubVector(match("a.txt", 0, 0.90, "a 的正文"), match("b.txt", 0, 0.70, "b 的正文"));

        List<RetrievedSegment> result = service.retrieve("问题");

        assertThat(result).hasSize(2);
        assertThat(result).allSatisfy(segment -> {
            assertThat(segment.rerankScore()).isNull();
            assertThat(segment.rrfScore()).isNull();
        });

        ArgumentCaptor<EmbeddingSearchRequest> captor = ArgumentCaptor.forClass(EmbeddingSearchRequest.class);
        verify(embeddingStore).search(captor.capture());
        assertThat(captor.getValue().maxResults()).isEqualTo(properties.getFinalTopK());
        assertThat(captor.getValue().minScore()).isEqualTo(properties.getMinScore());

        verify(keywordSearcher, never()).search(anyString(), anyInt(), anyDouble());
        verify(rerankClient, never()).rerank(anyString(), anyList());
    }

    @Test
    @DisplayName("粗召回为空（两个通道都没过闸）：直接返回空，不再浪费一次精排调用")
    void emptyCoarseRecallSkipsRerank() {
        properties.setHybridEnabled(true);
        properties.setRerankEnabled(true);
        when(embeddingStore.search(any(EmbeddingSearchRequest.class))).thenReturn(new EmbeddingSearchResult<>(List.of()));
        stubKeyword();

        assertThat(service.retrieve("今天天气怎么样")).isEmpty();
        verify(rerankClient, never()).rerank(anyString(), anyList());
    }

    @Test
    @DisplayName("最终喂给大模型的条数不超过 finalTopK")
    void resultIsTruncatedToFinalTopK() {
        properties.setRerankEnabled(false);
        properties.setFinalTopK(2);
        stubVector(match("a.txt", 0, 0.90, "a"), match("b.txt", 0, 0.80, "b"), match("c.txt", 0, 0.70, "c"));

        assertThat(service.retrieve("问题")).hasSize(2);
    }
}
