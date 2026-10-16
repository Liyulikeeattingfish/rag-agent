package com.ragagent.bench;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * 向量索引压测：在同一批数据上对比「精确扫描 / HNSW / IVFFlat」的召回与延迟。
 *
 * 为什么不能只在小语料上测：本项目真实语料只有 22 段，加不加索引都是微秒级，
 * 看不出任何差别。要看索引是否值得，必须有足够大的数据量，所以这里用合成向量灌到万级。
 *
 * 合成向量为什么不是均匀随机的：
 *   1024 维空间里均匀随机的点几乎两两等距（维度灾难），任意近邻都"差不多近"，
 *   会让 ANN 的召回看起来假性变差，也会让精确扫描的耗时虚高。真实 embedding 的
 *   内在维度远低于 1024，所以这里用「聚类中心 + 高斯噪声」来模拟：先随机取 K 个中心，
 *   每条向量取一个中心加噪声再归一化。这样数据有明显邻域结构，测出来的召回才有参考价值。
 *
 * 精确扫描的结果就是 ground truth —— 它没有近似，召回按定义是 100%。
 */
@Service
public class VectorIndexBenchmark {

    private static final Logger log = LoggerFactory.getLogger(VectorIndexBenchmark.class);

    private static final int DIMENSION = 1024;
    private static final int INSERT_BATCH = 500;

    /**
     * 噪声强度，相对聚类中心的模长而言。
     *
     * 这里有个必须踩过才知道的坑：噪声要按「每个维度的标准差」来给，而不是给一个固定值。
     * 中心向量归一化后模长是 1，如果在 1024 个维度上各加 σ=0.3 的高斯噪声，
     * 噪声向量的模长约为 0.3 × √1024 ≈ 9.6 —— 中心被彻底淹没，数据退化成近似均匀随机，
     * 而均匀随机在 1024 维下几乎两两等距（维度灾难），ANN 的召回会假性暴跌。
     * 除以 √D 之后，噪声模长才真正是中心模长的 30%，聚类结构被保留。
     */
    private static final double DEFAULT_NOISE_RATIO = 0.3;

    private final DataSource dataSource;

    public VectorIndexBenchmark(DataSource dataSource) {
        this.dataSource = dataSource;
    }

    /**
     * @param rows     灌入的向量条数
     * @param queries  查询条数
     * @param topK     每次查询取 top K
     * @param clusters 合成数据的聚类中心个数
     */
    public Map<String, Object> run(int rows, int queries, int topK, int clusters) throws SQLException {
        return run(rows, queries, topK, clusters, DEFAULT_NOISE_RATIO);
    }

    /**
     * @param noiseRatio 合成数据的噪声强度（相对聚类中心模长）。调大它数据更难区分，
     *                   HNSW 的召回损失才会显现出来 —— 默认 0.3 时这组数据上几乎测不出损失。
     */
    public Map<String, Object> run(int rows, int queries, int topK, int clusters, double noiseRatio) throws SQLException {
        long startAll = System.currentTimeMillis();
        Random random = new Random(20261016L);   // 固定种子，保证结果可复现

        float[][] centers = randomCenters(clusters, random);
        String[] queryLiterals = new String[queries];
        for (int i = 0; i < queries; i++) {
            queryLiterals[i] = toLiteral(sample(centers, random, noiseRatio));
        }

        try (Connection connection = dataSource.getConnection()) {
            long insertStart = System.currentTimeMillis();
            resetTable(connection);
            insertRows(connection, rows, centers, random, noiseRatio);
            long insertMillis = System.currentTimeMillis() - insertStart;
            log.info("[bench] 灌入 {} 条 {} 维向量，耗时 {} ms", rows, DIMENSION, insertMillis);

            List<Map<String, Object>> results = new ArrayList<>();

            // 1) 精确扫描：此时表上没有任何索引，结果即 ground truth
            long[][] truth = exactSearch(connection, queryLiterals, topK);
            Map<String, Object> exact = measure(connection, "精确扫描（无索引）", "无", queryLiterals, truth, topK, false);
            exact.put("buildMillis", 0L);
            exact.put("indexBytes", 0L);
            results.add(exact);

            // 2) HNSW：ef_search 是查询期参数，可以在线调
            long buildStart = System.currentTimeMillis();
            connection.createStatement().execute(
                    "CREATE INDEX bench_hnsw ON bench_vectors USING hnsw (embedding vector_cosine_ops)"
                            + " WITH (m = 16, ef_construction = 64)");
            long hnswBuild = System.currentTimeMillis() - buildStart;
            long hnswBytes = indexBytes(connection, "bench_hnsw");
            results.add(hnswResult(connection, queryLiterals, truth, topK, "HNSW ef_search=40", hnswBuild, hnswBytes, 40));
            results.add(hnswResult(connection, queryLiterals, truth, topK, "HNSW ef_search=100", 0L, hnswBytes, 100));

            // 对照组：同样的索引和参数，但把查询向量直接内联进 SQL，绕开服务端预编译语句。
            // 加这一组是因为踩到了一个真实的坑 —— 见 measure() 里的说明。
            Map<String, Object> hnswInlined = measure(connection, "HNSW", "ef_search=40，字面量内联",
                    queryLiterals, truth, topK, true);
            hnswInlined.put("label", "HNSW ef_search=40（内联字面量，不走服务端预编译）");
            hnswInlined.put("buildMillis", hnswBuild);
            hnswInlined.put("indexBytes", hnswBytes);
            results.add(hnswInlined);

            connection.createStatement().execute("DROP INDEX bench_hnsw");

            // 3) IVFFlat：注意它必须建在已有数据的表上，否则簇划分是退化的
            long ivfBuildStart = System.currentTimeMillis();
            int lists = Math.max(1, rows / 1000);
            connection.createStatement().execute(
                    "CREATE INDEX bench_ivfflat ON bench_vectors USING ivfflat (embedding vector_cosine_ops)"
                            + " WITH (lists = " + lists + ")");
            long ivfBuild = System.currentTimeMillis() - ivfBuildStart;
            long ivfBytes = indexBytes(connection, "bench_ivfflat");
            results.add(ivfResult(connection, queryLiterals, truth, topK, lists, ivfBuild, ivfBytes));
            connection.createStatement().execute("DROP INDEX bench_ivfflat");

            // 4) 直接把表里的数据交给 pgvector 统计，确认索引体积口径
            long tableBytes = indexBytes(connection, "bench_vectors");

            Map<String, Object> response = new LinkedHashMap<>();
            response.put("rows", rows);
            response.put("dimension", DIMENSION);
            response.put("queries", queries);
            response.put("topK", topK);
            response.put("clusters", clusters);
            response.put("noiseRatio", noiseRatio);
            response.put("insertMillis", insertMillis);
            response.put("tableBytes", tableBytes);
            response.put("totalMillis", System.currentTimeMillis() - startAll);
            response.put("results", results);
            return response;
        }
    }

    // ------------------------------------------------------------------
    // 数据准备
    // ------------------------------------------------------------------

    private void resetTable(Connection connection) throws SQLException {
        connection.createStatement().execute("DROP TABLE IF EXISTS bench_vectors");
        connection.createStatement().execute(
                "CREATE TABLE bench_vectors (id bigserial PRIMARY KEY, embedding vector(" + DIMENSION + "))");
    }

    private void insertRows(Connection connection, int rows, float[][] centers, Random random, double noiseRatio)
            throws SQLException {
        connection.setAutoCommit(false);
        try (PreparedStatement ps = connection.prepareStatement("INSERT INTO bench_vectors(embedding) VALUES (?::vector)")) {
            for (int i = 0; i < rows; i++) {
                ps.setString(1, toLiteral(sample(centers, random, noiseRatio)));
                ps.addBatch();
                if ((i + 1) % INSERT_BATCH == 0) {
                    ps.executeBatch();
                }
            }
            ps.executeBatch();
        }
        connection.commit();
        connection.setAutoCommit(true);
        // 批量导入后必须 ANALYZE：否则计划器拿不到真实的行数统计，
        // 明明建了 HNSW 索引它也可能选顺序扫描 —— 那测出来的就不是索引的性能了。
        connection.createStatement().execute("ANALYZE bench_vectors");
    }

    private float[][] randomCenters(int clusters, Random random) {
        float[][] centers = new float[clusters][DIMENSION];
        for (float[] center : centers) {
            for (int i = 0; i < DIMENSION; i++) {
                center[i] = (float) random.nextGaussian();
            }
            normalize(center);
        }
        return centers;
    }

    /**
     * 取一个聚类中心，加上按维度归一化后的高斯噪声。
     *
     * @param noiseRatio 噪声模长相对于中心模长的比例（0.3 表示噪声占三成）
     */
    private float[] sample(float[][] centers, Random random, double noiseRatio) {
        float[] center = centers[random.nextInt(centers.length)];
        double sigma = noiseRatio / Math.sqrt(DIMENSION);
        float[] vector = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            vector[i] = (float) (center[i] + random.nextGaussian() * sigma);
        }
        normalize(vector);
        return vector;
    }

    private static void normalize(float[] vector) {
        double sum = 0;
        for (float v : vector) {
            sum += (double) v * v;
        }
        double norm = Math.sqrt(sum);
        if (norm == 0) {
            return;
        }
        for (int i = 0; i < vector.length; i++) {
            vector[i] = (float) (vector[i] / norm);
        }
    }

    /** pgvector 的字面量格式：[0.1,0.2,...] */
    private static String toLiteral(float[] vector) {
        StringBuilder sb = new StringBuilder(vector.length * 12 + 2);
        sb.append('[');
        for (int i = 0; i < vector.length; i++) {
            if (i > 0) {
                sb.append(',');
            }
            sb.append(vector[i]);
        }
        sb.append(']');
        return sb.toString();
    }

    // ------------------------------------------------------------------
    // 测量
    // ------------------------------------------------------------------

    /** 无索引时的精确检索，结果作为 ground truth。 */
    private long[][] exactSearch(Connection connection, String[] queryLiterals, int topK) throws SQLException {
        String sql = "SELECT id FROM bench_vectors ORDER BY embedding <=> ?::vector LIMIT " + topK;
        long[][] truth = new long[queryLiterals.length][];
        try (PreparedStatement ps = connection.prepareStatement(sql)) {
            for (int q = 0; q < queryLiterals.length; q++) {
                ps.setString(1, queryLiterals[q]);
                try (ResultSet rs = ps.executeQuery()) {
                    truth[q] = collectIds(rs, topK);
                }
            }
        }
        return truth;
    }

    private Map<String, Object> hnswResult(Connection connection, String[] queryLiterals, long[][] truth,
                                           int topK, String label, long buildMillis, long indexBytes,
                                           int efSearch) throws SQLException {
        connection.createStatement().execute("SET hnsw.ef_search = " + efSearch);
        Map<String, Object> result = measure(connection, "HNSW", "ef_search=" + efSearch,
                queryLiterals, truth, topK, false);
        result.put("label", label);
        result.put("buildMillis", buildMillis);
        result.put("indexBytes", indexBytes);
        return result;
    }

    private Map<String, Object> ivfResult(Connection connection, String[] queryLiterals, long[][] truth,
                                          int topK, int lists, long buildMillis, long indexBytes) throws SQLException {
        int probes = Math.max(1, (int) Math.round(Math.sqrt(lists)));
        connection.createStatement().execute("SET ivfflat.probes = " + probes);
        Map<String, Object> result = measure(connection, "IVFFlat", "lists=" + lists + ", probes=" + probes,
                queryLiterals, truth, topK, false);
        result.put("label", "IVFFlat lists=" + lists + " probes=" + probes);
        result.put("buildMillis", buildMillis);
        result.put("indexBytes", indexBytes);
        return result;
    }

    /**
     * 跑一组查询并统计召回与延迟。
     *
     * @param inlined true 表示把查询向量直接拼进 SQL（普通 Statement，每次都是一次性计划）；
     *                false 表示走 PreparedStatement 绑定参数。
     *
     * ★ 这个参数是为了解释一个真实踩到的坑 ★
     * 一开始只用了 PreparedStatement，结果出现了非常反直觉的现象：EXPLAIN ANALYZE 显示
     * 服务端只花 0.6 ms 走了 HNSW 索引，但客户端测出来每次都要 44 ms —— 和全表顺序扫描一模一样。
     * 原因是 pgjdbc 的 prepareThreshold 默认为 5：同一个语句执行到第 6 次时，驱动会切换到
     * **服务端预编译语句 + 通用计划（generic plan）**。通用计划下计划器拿不到参数的具体值，
     * 对 pgvector 的索引代价估算会偏保守，于是放弃索引改走顺序扫描 —— 索引白建了，而且不报任何错。
     * 所以这里同时测「绑定参数」和「内联字面量」两种写法，把差别摆在明面上。
     */
    private Map<String, Object> measure(Connection connection, String index, String params, String[] queryLiterals,
                                        long[][] truth, int topK, boolean inlined) throws SQLException {
        String sql = "SELECT id FROM bench_vectors ORDER BY embedding <=> ?::vector LIMIT " + topK;
        String inlinedTemplate = "SELECT id FROM bench_vectors ORDER BY embedding <=> '%s'::vector LIMIT " + topK;

        // 预热：第一次执行会带来连接、计划缓存等一次性开销，不能算进延迟
        for (int i = 0; i < Math.min(5, queryLiterals.length); i++) {
            runOne(connection, sql, queryLiterals[i], topK);
        }

        long[] latencies = new long[queryLiterals.length];
        double recallSum = 0;

        // 绑定参数时复用同一个 PreparedStatement；内联时每次都是新语句（因此每次都是自定义计划）
        PreparedStatement ps = inlined ? null : connection.prepareStatement(sql);
        try {
            for (int q = 0; q < queryLiterals.length; q++) {
                long t0 = System.nanoTime();
                long[] ids;
                if (inlined) {
                    try (Statement statement = connection.createStatement();
                         ResultSet rs = statement.executeQuery(String.format(inlinedTemplate, queryLiterals[q]))) {
                        ids = collectIds(rs, topK);
                    }
                } else {
                    ps.setString(1, queryLiterals[q]);
                    try (ResultSet rs = ps.executeQuery()) {
                        ids = collectIds(rs, topK);
                    }
                }
                latencies[q] = (System.nanoTime() - t0) / 1_000_000;

                int hit = 0;
                for (long id : ids) {
                    for (long expected : truth[q]) {
                        if (id == expected) {
                            hit++;
                            break;
                        }
                    }
                }
                recallSum += (double) hit / Math.min(topK, truth[q].length);
            }
        } finally {
            if (ps != null) {
                ps.close();
            }
        }
        Arrays.sort(latencies);

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("index", index);
        result.put("params", params);
        result.put("recallAtK", round4(recallSum / queryLiterals.length));
        result.put("serverMillis", 0.0);
        result.put("minMillis", latencies[0]);
        result.put("p10Millis", percentile(latencies, 0.10));
        result.put("p50Millis", median(latencies));
        result.put("p95Millis", percentile(latencies, 0.95));
        result.put("maxMillis", latencies[latencies.length - 1]);
        // 把执行计划和服务端耗时一并带出来：跑得快是因为走了索引，还是因为计划器选了顺序扫描，
        // 光看客户端延迟分不出来。这是"测的到底是不是索引"的凭证。
        result.put("plan", explainPlan(connection, queryLiterals[0], topK));
        result.put("serverMillis", explainAnalyzeMillis(connection, queryLiterals[0], topK));
        result.put("inlined", inlined);
        return result;
    }

    private long[] runOne(Connection connection, String sql, String literal, int topK) throws SQLException {
        try (PreparedStatement ps = connection.prepareStatement(sql)) {
            ps.setString(1, literal);
            try (ResultSet rs = ps.executeQuery()) {
                return collectIds(rs, topK);
            }
        }
    }

    private static long[] collectIds(ResultSet rs, int topK) throws SQLException {
        long[] ids = new long[topK];
        int n = 0;
        while (rs.next() && n < topK) {
            ids[n++] = rs.getLong(1);
        }
        return n == topK ? ids : Arrays.copyOf(ids, n);
    }

    /**
     * 精简后的执行计划：只保留"走了哪种扫描、有没有排序"这类结论性的行。
     *
     * 为什么要裁剪：EXPLAIN 会把 1024 维查询向量整个原样打印出来，一条计划几万字符，
     * 打到日志里既看不清也没意义。
     */
    private String explainPlan(Connection connection, String literal, int topK) {
        String sql = "EXPLAIN SELECT id FROM bench_vectors ORDER BY embedding <=> ?::vector LIMIT " + topK;
        try (PreparedStatement ps = connection.prepareStatement(sql)) {
            ps.setString(1, literal);
            try (ResultSet rs = ps.executeQuery()) {
                StringBuilder sb = new StringBuilder();
                while (rs.next()) {
                    String line = rs.getString(1).replaceAll("'\\[[^]]*]'::vector", "<query-vector>").trim();
                    // 只留结论性的节点，跳过带有大量数值成本的细节行
                    if (line.contains("Scan") || line.contains("Sort") || line.startsWith("Limit")) {
                        if (sb.length() > 0) {
                            sb.append(" / ");
                        }
                        sb.append(line.replaceAll("\\(cost=[^)]*\\)", "").replaceAll("\\s+", " ").trim());
                    }
                }
                return sb.length() == 0 ? "（未取到扫描节点）" : sb.toString();
            }
        } catch (SQLException e) {
            return "EXPLAIN 失败：" + e.getMessage();
        }
    }

    /** 服务端真实执行时间（EXPLAIN ANALYZE 的 Execution Time），用来校准客户端测出来的延迟。 */
    private double explainAnalyzeMillis(Connection connection, String literal, int topK) {
        String sql = "EXPLAIN (ANALYZE, TIMING OFF, SUMMARY ON) "
                + "SELECT id FROM bench_vectors ORDER BY embedding <=> ?::vector LIMIT " + topK;
        try (PreparedStatement ps = connection.prepareStatement(sql)) {
            ps.setString(1, literal);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    String line = rs.getString(1);
                    int at = line.indexOf("Execution Time:");
                    if (at >= 0) {
                        return Double.parseDouble(line.substring(at + "Execution Time:".length()).replace("ms", "").trim());
                    }
                }
            }
        } catch (Exception e) {
            return -1;
        }
        return -1;
    }

    private long indexBytes(Connection connection, String name) {
        try (ResultSet rs = connection.createStatement().executeQuery(
                "SELECT pg_relation_size('" + name + "')")) {
            return rs.next() ? rs.getLong(1) : 0L;
        } catch (SQLException e) {
            return 0L;
        }
    }

    private static double median(long[] sorted) {
        return sorted[sorted.length / 2];
    }

    private static double percentile(long[] sorted, double p) {
        int index = (int) Math.ceil(p * sorted.length) - 1;
        return sorted[Math.max(0, Math.min(index, sorted.length - 1))];
    }

    private static double round4(double value) {
        return Math.round(value * 10000) / 10000.0;
    }
}
