package com.ragagent.rag;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * BM25 通道的单测：分词、打分排序、以及缓存失效。
 *
 * 打分是纯计算，最值得钉住 —— 权重算错不会报错，只会让关键词通道"看起来在工作但排得不对"。
 */
@ExtendWith(MockitoExtension.class)
class Bm25KeywordSearcherTest {

    @Mock
    private JdbcTemplate jdbcTemplate;

    private Bm25KeywordSearcher searcher;

    @BeforeEach
    void setUp() {
        searcher = new Bm25KeywordSearcher(jdbcTemplate);
    }

    @SuppressWarnings("unchecked")
    private void stubCorpus(Object[]... rows) {
        when(jdbcTemplate.query(anyString(), any(RowMapper.class))).thenReturn(List.of(rows));
    }

    private static Object[] row(String text, String fileName, int index) {
        return new Object[]{text, fileName, index};
    }

    // ------------------------------------------------------------------
    // 分词
    // ------------------------------------------------------------------

    @Test
    @DisplayName("分词：ASCII 连续段成词，中文按二元组切，单字保留")
    void tokenizesAsciiWordsAndCjkBigrams() {
        assertThat(Bm25KeywordSearcher.tokenize("HNSW 索引"))
                .containsExactly("hnsw", "索引");
        assertThat(Bm25KeywordSearcher.tokenize("分层图索引"))
                .containsExactly("分层", "层图", "图索", "索引");
        assertThat(Bm25KeywordSearcher.tokenize("宽"))
                .containsExactly("宽");
        assertThat(Bm25KeywordSearcher.tokenize("ef_search=40"))
                .containsExactly("ef", "search", "40");
        assertThat(Bm25KeywordSearcher.tokenize("  ")).isEmpty();
    }

    // ------------------------------------------------------------------
    // 打分与排序
    // ------------------------------------------------------------------

    @Test
    @DisplayName("查询词命中谁，谁就排前面")
    void ranksTheMatchingSegmentFirst() {
        stubCorpus(
                row("HNSW 是分层图索引，查询从上层入口点开始逐层下沉", "hnsw.md", 0),
                row("IVFFlat 是聚类索引，查询只扫描最近的若干个簇", "ivfflat.md", 0));

        List<KeywordHit> hits = searcher.search("IVFFlat 簇", 10, 0.0);

        assertThat(hits).isNotEmpty();
        assertThat(hits.get(0).fileName()).isEqualTo("ivfflat.md");
        assertThat(hits.get(0).score()).isGreaterThan(0);
    }

    @Test
    @DisplayName("IDF 起作用：稀有词比通用词权重高")
    void rareTermsWeighMoreThanCommonOnes() {
        // "检索" 在两份文档里都出现（df=2），"BM25" 只在一份里出现（df=1）
        stubCorpus(
                row("检索通道 BM25 会计算词频", "bm25.md", 0),
                row("检索通道 pg_trgm 只看三元组", "trgm.md", 0));

        List<KeywordHit> hits = searcher.search("BM25 检索", 10, 0.0);

        assertThat(hits.get(0).fileName()).isEqualTo("bm25.md");
        // 只命中通用词的文档也会被召回，但分数更低
        assertThat(hits.get(1).fileName()).isEqualTo("trgm.md");
        assertThat(hits.get(0).score()).isGreaterThan(hits.get(1).score());
    }

    @Test
    @DisplayName("完全不相关的查询返回空，而不是硬凑几条")
    void returnsEmptyWhenNoTermMatches() {
        stubCorpus(row("HNSW 是分层图索引", "hnsw.md", 0));

        assertThat(searcher.search("今天天气怎么样", 10, 0.0)).isEmpty();
    }

    @Test
    @DisplayName("limit 生效")
    void respectsLimit() {
        stubCorpus(
                row("索引 A", "a.md", 0),
                row("索引 B", "b.md", 0),
                row("索引 C", "c.md", 0));

        assertThat(searcher.search("索引", 2, 0.0)).hasSize(2);
    }

    // ------------------------------------------------------------------
    // 缓存
    // ------------------------------------------------------------------

    @Test
    @DisplayName("索引惰性构建且被缓存；invalidate 之后重建（否则新入库的文档永远搜不到）")
    void cachesIndexUntilInvalidated() {
        stubCorpus(row("HNSW 是分层图索引", "hnsw.md", 0));

        searcher.search("分层", 10, 0.0);
        searcher.search("索引", 10, 0.0);
        verify(jdbcTemplate, times(1)).query(anyString(), any(RowMapper.class));

        searcher.invalidate();
        searcher.search("分层", 10, 0.0);
        verify(jdbcTemplate, times(2)).query(anyString(), any(RowMapper.class));
    }
}
