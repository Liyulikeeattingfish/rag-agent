package com.ragagent.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * 检索相关的全部参数。
 *
 * 为什么要收拢到一个类里：这些值同时被「粗召回」「RRF 融合」「精排」三段逻辑使用，
 * 散落成一堆 @Value 之后，改一个参数要在三四个文件里找。收进一个 @ConfigurationProperties
 * 还顺带得到两个好处：IDE 里有补全提示，且能在启动参数里整体覆盖
 * （如 --rag.retrieval.rerank-enabled=false）。
 *
 * 覆盖优先级（从低到高）：本类里的字段默认值 → application.yml → 环境变量 → 命令行参数。
 * 也就是说：即使 yml 里一个都没写，程序也能按下面的默认值跑起来。
 */
@Component
@ConfigurationProperties(prefix = "rag.retrieval")
public class RetrievalProperties {

    /**
     * 向量通道的相似度下限。
     * 注意 LangChain4j 的 pgvector 实现里 score = (1 + 余弦相似度) / 2，
     * 所以 0.66 实际约等于「余弦相似度 >= 0.32」，本项目语料实测相关段 0.77~0.84、
     * 不相关段 0.61~0.62，0.66 正好卡在中间。换语料要重新标定。
     *
     * 这个值现在是「粗召回的第一道闸」：向量通道低于它的段落直接不进候选集，
     * 这也是「资料里没有就问什么都能答」被挡住的原因。
     */
    private double minScore = 0.66;

    /** 向量通道粗召回条数（召回候选的宽度）。 */
    private int vectorTopK = 10;

    /** 关键词（pg_trgm）通道粗召回条数。 */
    private int keywordTopK = 10;

    /**
     * 关键词通道的 trigram 相似度下限。
     *
     * 默认 0.0，含义是「严格大于 0」——即要求查询与片段至少共享一个字符三元组。
     * 为什么不能设高：pg_trgm 的 similarity = 共享三元组数 / 三元组并集数，
     * 拿一个 5 字的中文短查询去比一个 400 字的片段，就算查询完全被片段包含，
     * 分数也只有 5/400 ≈ 0.01 量级。阈值设 0.05 会把所有正常命中全部误杀。
     * 所以这个通道的价值在「有没有交集」，不在「分数绝对值」。
     */
    private double keywordMinSimilarity = 0.0;

    /** RRF 融合后保留的候选条数，也就是交给 rerank 的条数（粗召回候选集大小）。 */
    private int candidateTopK = 10;

    /** 最终喂给大模型的段落数。 */
    private int finalTopK = 3;

    /**
     * RRF 公式里的常数 k：score(d) = Σ 1 / (k + rank)。
     * 60 是原论文的取值，业界默认沿用它。k 越大，排名靠后的候选被拉开得越平缓。
     */
    private int rrfK = 60;

    /** 是否启用关键词通道参与粗召回。关掉即退回纯向量召回。 */
    private boolean hybridEnabled = true;

    /** 是否启用 rerank 精排。 */
    private boolean rerankEnabled = true;

    /** rerank 模型名。 */
    private String rerankModel = "BAAI/bge-reranker-v2-m3";

    /**
     * rerank 分数下限。低于它的片段直接丢弃，不再喂给大模型；候选全被丢弃即视为拒答。
     *
     * 0.05 不是拍脑袋定的，是本项目语料实测标定出来的（20 条评测用例，见 .workbuddy/eval/）：
     *   相关用例的 rank1 分数落在 0.17 ~ 0.9993，
     *   无关用例（今天天气 / 推荐电影 / 炒西红柿鸡蛋 / Python 的 GIL）最高只有 0.0014。
     *   0.05 正好落在这条 0.0014 ~ 0.17 的空档里，两端都留了余量（下方 35 倍、上方 3.5 倍）。
     *
     * 为什么拒答不能只靠粗召回的 min-score：那是向量分的闸门，实测拦不住
     * 「Python 的 GIL 是什么」这类问题——语料里也有「Python」「向量」这些字眼，
     * 向量分能过 0.66 的闸。而 rerank 能把它压到 0.0014。两道闸拦的是两类不同的东西，
     * 缺一不可：min-score 挡「语义上就不沾边」的，rerank-min-score 挡「语义像但答非所问」的。
     *
     * 反过来更要小心：rerank 只排序、不拒答。这个阈值设回 0.0 时，
     * 实测拒答率会从 100% 直接掉到 50%（粗召回候选池比纯向量 top3 更大，噪声更容易挤进前 3）。
     *
     * 换语料 / 换 rerank 模型必须重新标定这个值。
     */
    private double rerankMinScore = 0.05;

    public double getMinScore() {
        return minScore;
    }

    public void setMinScore(double minScore) {
        this.minScore = minScore;
    }

    public int getVectorTopK() {
        return vectorTopK;
    }

    public void setVectorTopK(int vectorTopK) {
        this.vectorTopK = vectorTopK;
    }

    public int getKeywordTopK() {
        return keywordTopK;
    }

    public void setKeywordTopK(int keywordTopK) {
        this.keywordTopK = keywordTopK;
    }

    public double getKeywordMinSimilarity() {
        return keywordMinSimilarity;
    }

    public void setKeywordMinSimilarity(double keywordMinSimilarity) {
        this.keywordMinSimilarity = keywordMinSimilarity;
    }

    public int getCandidateTopK() {
        return candidateTopK;
    }

    public void setCandidateTopK(int candidateTopK) {
        this.candidateTopK = candidateTopK;
    }

    public int getFinalTopK() {
        return finalTopK;
    }

    public void setFinalTopK(int finalTopK) {
        this.finalTopK = finalTopK;
    }

    public int getRrfK() {
        return rrfK;
    }

    public void setRrfK(int rrfK) {
        this.rrfK = rrfK;
    }

    public boolean isHybridEnabled() {
        return hybridEnabled;
    }

    public void setHybridEnabled(boolean hybridEnabled) {
        this.hybridEnabled = hybridEnabled;
    }

    public boolean isRerankEnabled() {
        return rerankEnabled;
    }

    public void setRerankEnabled(boolean rerankEnabled) {
        this.rerankEnabled = rerankEnabled;
    }

    public String getRerankModel() {
        return rerankModel;
    }

    public void setRerankModel(String rerankModel) {
        this.rerankModel = rerankModel;
    }

    public double getRerankMinScore() {
        return rerankMinScore;
    }

    public void setRerankMinScore(double rerankMinScore) {
        this.rerankMinScore = rerankMinScore;
    }
}
