package com.ragagent.rag;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

/**
 * 关键词通道之二：BM25。
 *
 * 和 pg_trgm 的区别不是"都算关键词所以差不多"，而是打分的机理完全不同：
 *   pg_trgm  算的是两个字符串共享多少字符三元组，**没有词频、没有逆文档频率、没有长度归一化**。
 *            好处是中文免分词，代价是"常见字组合"和"罕见专有名词"权重一样，长文本天然占便宜。
 *   BM25     是词级打分：IDF 压低高频词、词频饱和让"出现 10 次"不再比"出现 5 次"强一倍、
 *            长度归一化惩罚长文档。它更准，但中文必须先分词。
 *
 * 中文分词怎么解决：不引第三方分词器，用**字符二元组（bigram）**近似。这是免依赖的折中 ——
 * 和 pg_trgm 用三元组是同一个思路，只是粒度更细、并且补上了词频与 IDF。
 * 代价是看不见词边界，对同义词与词形变化无能为力，这一点必须说清楚。
 *
 * 索引为什么要缓存：
 *   BM25 的 IDF 需要全局的文档频率，每次查询都把整表读出来算一遍，语料一大就是灾难。
 *   所以这里在内存里维护倒排索引，并在语料变更时由 DocumentService 调用 invalidate() 作废。
 *   这是典型的"读写权衡"：牺牲一点内存和一次重建，换查询期 O(词数 × 平均倒排长度)。
 *   注意副作用：如果有人绕过接口直接往表里写数据，缓存不会失效 —— 这是个已知边界。
 *
 * 上线方式：--rag.retrieval.keyword-channel=bm25
 */
@Component
@ConditionalOnProperty(prefix = "rag.retrieval", name = "keyword-channel", havingValue = "bm25")
public class Bm25KeywordSearcher implements KeywordChannel {

    /** BM25 的两个经验参数：k1 控制词频饱和速度，b 控制文档长度归一化的强度。 */
    private static final double K1 = 1.2;
    private static final double B = 0.75;

    private static final String TABLE = "rag_segments";

    private final JdbcTemplate jdbcTemplate;
    private final Object lock = new Object();

    /** volatile + 双重检查：查询路径读多写少，不想每次查询都抢锁。 */
    private volatile CorpusIndex cached;

    public Bm25KeywordSearcher(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    @Override
    public List<KeywordHit> search(String query, int limit, double minScore) {
        List<String> queryTerms = tokenize(query);
        if (queryTerms.isEmpty()) {
            return List.of();
        }
        // BM25 的分数没有跨语料的绝对阈值意义（不像 trigram 相似度有 0~1 的固定含义），
        // 所以这里只要求"至少命中一个查询词"，minScore 不参与过滤。
        return index().search(queryTerms, limit);
    }

    @Override
    public void invalidate() {
        cached = null;
    }

    private CorpusIndex index() {
        CorpusIndex snapshot = cached;
        if (snapshot != null) {
            return snapshot;
        }
        synchronized (lock) {
            if (cached == null) {
                cached = CorpusIndex.load(jdbcTemplate, TABLE);
            }
            return cached;
        }
    }

    // ------------------------------------------------------------------
    // 分词
    // ------------------------------------------------------------------

    /**
     * 极简分词：ASCII 字母数字按连续段切词，CJK 按字符二元组切，单字成词时保留单字。
     * 没有引分词器，也没有去停用词 —— 高频词的权重交给 IDF 压。
     */
    static List<String> tokenize(String text) {
        List<String> terms = new ArrayList<>();
        if (text == null || text.isBlank()) {
            return terms;
        }
        StringBuilder word = new StringBuilder();
        List<Character> cjkRun = new ArrayList<>();

        for (char c : text.toLowerCase().toCharArray()) {
            if (c < 128 && Character.isLetterOrDigit(c)) {
                flushCjk(cjkRun, terms);
                word.append(c);
                continue;
            }
            if (isCjk(c)) {
                flushWord(word, terms);
                cjkRun.add(c);
                continue;
            }
            flushWord(word, terms);
            flushCjk(cjkRun, terms);
        }
        flushWord(word, terms);
        flushCjk(cjkRun, terms);
        return terms;
    }

    private static void flushWord(StringBuilder word, List<String> terms) {
        if (word.length() > 0) {
            terms.add(word.toString());
            word.setLength(0);
        }
    }

    private static void flushCjk(List<Character> run, List<String> terms) {
        if (run.isEmpty()) {
            return;
        }
        if (run.size() == 1) {
            terms.add(String.valueOf(run.get(0)));
        } else {
            for (int i = 0; i + 1 < run.size(); i++) {
                terms.add("" + run.get(i) + run.get(i + 1));
            }
        }
        run.clear();
    }

    private static boolean isCjk(char c) {
        return c >= 0x4E00 && c <= 0x9FFF;
    }

    // ------------------------------------------------------------------
    // 倒排索引
    // ------------------------------------------------------------------

    /** 一个片段的原文与身份。 */
    private record Segment(String text, String fileName, Integer segmentIndex, int length) {
    }

    private static final class CorpusIndex {

        private final List<Segment> segments;
        private final Map<String, List<int[]>> postings;   // term -> [segmentIndex, termFrequency]
        private final double averageLength;

        private CorpusIndex(List<Segment> segments, Map<String, List<int[]>> postings, double averageLength) {
            this.segments = segments;
            this.postings = postings;
            this.averageLength = averageLength;
        }

        static CorpusIndex load(JdbcTemplate jdbcTemplate, String table) {
            List<Object[]> rows = jdbcTemplate.query(
                    "SELECT text, metadata->>'file_name' AS file_name, "
                            + "NULLIF(metadata->>'index', '')::int AS segment_index FROM " + table,
                    (rs, rowNum) -> new Object[]{rs.getString("text"),
                            rs.getString("file_name"), rs.getObject("segment_index")});

            List<Segment> segments = new ArrayList<>(rows.size());
            Map<String, List<int[]>> postings = new HashMap<>();
            long totalLength = 0;

            for (int i = 0; i < rows.size(); i++) {
                Object[] row = rows.get(i);
                String text = (String) row[0];
                List<String> terms = tokenize(text);
                Map<String, Integer> frequency = new HashMap<>();
                for (String term : terms) {
                    frequency.merge(term, 1, Integer::sum);
                }
                segments.add(new Segment(text, (String) row[1], (Integer) row[2], terms.size()));
                totalLength += terms.size();
                for (Map.Entry<String, Integer> entry : frequency.entrySet()) {
                    postings.computeIfAbsent(entry.getKey(), key -> new ArrayList<>())
                            .add(new int[]{i, entry.getValue()});
                }
            }
            double averageLength = segments.isEmpty() ? 1.0
                    : Math.max(1.0, (double) totalLength / segments.size());
            return new CorpusIndex(segments, postings, averageLength);
        }

        List<KeywordHit> search(List<String> queryTerms, int limit) {
            if (segments.isEmpty()) {
                return List.of();
            }
            int totalDocs = segments.size();
            Map<Integer, Double> scores = new HashMap<>();

            for (String term : new LinkedHashSet<>(queryTerms)) {
                List<int[]> posted = postings.get(term);
                if (posted == null) {
                    continue;
                }
                int documentFrequency = posted.size();
                double idf = Math.log(1 + (totalDocs - documentFrequency + 0.5) / (documentFrequency + 0.5));
                for (int[] entry : posted) {
                    int segmentIndex = entry[0];
                    int termFrequency = entry[1];
                    double length = segments.get(segmentIndex).length();
                    double denominator = termFrequency + K1 * (1 - B + B * length / averageLength);
                    scores.merge(segmentIndex, idf * (termFrequency * (K1 + 1)) / denominator, Double::sum);
                }
            }
            if (scores.isEmpty()) {
                return List.of();
            }

            List<Map.Entry<Integer, Double>> ranked = new ArrayList<>(scores.entrySet());
            ranked.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));

            List<KeywordHit> hits = new ArrayList<>();
            for (Map.Entry<Integer, Double> entry : ranked) {
                if (hits.size() >= limit) {
                    break;
                }
                Segment segment = segments.get(entry.getKey());
                hits.add(new KeywordHit(segment.text(), segment.fileName(), segment.segmentIndex(), entry.getValue()));
            }
            return hits;
        }
    }

    /** 仅供测试查看分词结果的入口。 */
    static List<String> tokensOf(String text) {
        return Arrays.asList(tokenize(text).toArray(new String[0]));
    }
}
