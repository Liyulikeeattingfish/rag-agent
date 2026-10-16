package com.ragagent.eval;

/**
 * 一条评测用例。
 *
 * @param query        问题
 * @param expectedFile 期望被排进 top-k 的文档文件名；无关用例为 null
 * @param shouldAnswer true = 语料里有答案；false = 无关问题，期望系统拒答
 */
public record EvalCase(String query, String expectedFile, boolean shouldAnswer) {
}
