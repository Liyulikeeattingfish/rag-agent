package com.ragagent.rag;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/** 指标口径的单测：算错了不会报错，只会让你基于错误的数字做决策。 */
class RetrievalMetricsTest {

    @Test
    @DisplayName("rankOf 返回 1 开始的名次，未命中返回 0")
    void rankOfFindsPosition() {
        List<String> files = List.of("a.md", "b.md", "c.md");

        assertThat(RetrievalMetrics.rankOf(files, "a.md")).isEqualTo(1);
        assertThat(RetrievalMetrics.rankOf(files, "c.md")).isEqualTo(3);
        assertThat(RetrievalMetrics.rankOf(files, "zzz.md")).isZero();
        assertThat(RetrievalMetrics.rankOf(files, null)).isZero();
        assertThat(RetrievalMetrics.rankOf(List.of(), "a.md")).isZero();
    }

    @Test
    @DisplayName("RR：第 1 名 1.0、第 2 名 0.5、第 4 名 0.25、未命中 0")
    void reciprocalRankIsOneOverRank() {
        assertThat(RetrievalMetrics.reciprocalRank(1)).isEqualTo(1.0);
        assertThat(RetrievalMetrics.reciprocalRank(2)).isEqualTo(0.5);
        assertThat(RetrievalMetrics.reciprocalRank(4)).isEqualTo(0.25);
        assertThat(RetrievalMetrics.reciprocalRank(0)).isZero();
    }

    @Test
    @DisplayName("nDCG@3：第 1 名 1.0、第 2 名约 0.6309、第 3 名 0.5、第 4 名超出窗口记 0")
    void ndcgAtThreeMatchesDefinition() {
        assertThat(RetrievalMetrics.ndcgAtK(1, 3)).isEqualTo(1.0);
        assertThat(RetrievalMetrics.ndcgAtK(2, 3)).isCloseTo(0.6309, within(1e-4));
        assertThat(RetrievalMetrics.ndcgAtK(3, 3)).isCloseTo(0.5, within(1e-4));
        assertThat(RetrievalMetrics.ndcgAtK(4, 3)).isZero();
        assertThat(RetrievalMetrics.ndcgAtK(0, 3)).isZero();
    }

    @Test
    @DisplayName("nDCG 随 k 变化：同一个 rank，窗口越大分越高（或被排除）")
    void ndcgDependsOnWindow() {
        assertThat(RetrievalMetrics.ndcgAtK(3, 5)).isCloseTo(0.5, within(1e-4));
        assertThat(RetrievalMetrics.ndcgAtK(3, 2)).isZero();
    }
}
