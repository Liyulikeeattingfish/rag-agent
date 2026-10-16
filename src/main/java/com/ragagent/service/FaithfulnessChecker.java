package com.ragagent.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 答案忠实度检查：回答里的结论是不是真的能从检索到的资料里找到依据。
 *
 * 分两层，各管各的，缺一不可：
 *
 *   规则层（{@link #checkCitations}）—— 只做能确定的事：回答有没有标引用、引用的编号是否
 *   落在实际喂进去的片段范围内。它拦得住"编造不存在的编号"和"压根没标注"，
 *   但它看不懂语义，拦不住"标了 [1] 却说的和资料不一样"。
 *
 *   模型层（{@link #judge}）—— 把资料和答案一起交给另一个模型问"有没有资料支撑不了的断言"。
 *   这是 LLM-as-judge，有随机性也有成本，所以温度固定为 0，且只在生成侧评测里用。
 *
 * 为什么不能只做模型层：模型层既慢又可能判错，规则层是零成本的确定性检查，
 * 能先挡掉一大类问题（比如模型忘了标引用）。
 */
@Service
public class FaithfulnessChecker {

    private static final Logger log = LoggerFactory.getLogger(FaithfulnessChecker.class);

    /** 引用编号的形态：[1] [12]。 */
    private static final Pattern CITATION = Pattern.compile("\\[(\\d+)]");

    /** 单段资料送进判定 prompt 的长度上限，防止把上下文撑爆。 */
    private static final int MAX_SOURCE_CHARS = 800;

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final ChatModel chatModel;

    public FaithfulnessChecker(ChatModel chatModel) {
        this.chatModel = chatModel;
    }

    /**
     * 规则层：检查引用编号。
     *
     * @param sourceCount 实际喂给模型的片段数，合法的编号范围是 1..sourceCount
     */
    public CitationCheck checkCitations(String answer, int sourceCount) {
        if (answer == null || answer.isBlank() || sourceCount <= 0) {
            return new CitationCheck(false, List.of(), 0);
        }
        Set<Integer> valid = new LinkedHashSet<>();
        int invalid = 0;
        Matcher matcher = CITATION.matcher(answer);
        while (matcher.find()) {
            int index = Integer.parseInt(matcher.group(1));
            if (index >= 1 && index <= sourceCount) {
                valid.add(index);
            } else {
                invalid++;
            }
        }
        return new CitationCheck(!valid.isEmpty(), List.copyOf(valid), invalid);
    }

    /**
     * 模型层：判定答案是否忠于资料。
     *
     * @return faithful 为 null 表示判定失败（调用异常或返回不是合法 JSON）——
     *         这时必须单独计数，不能默认算作"忠实"，否则失败会被掩盖成好结果。
     */
    public FaithfulnessVerdict judge(String question, List<String> sources, String answer) {
        try {
            String raw = chatModel.chat(ChatRequest.builder()
                    .messages(UserMessage.from(buildJudgePrompt(question, sources, answer)))
                    .temperature(0.0)   // 判定任务不该有随机性
                    .build()).aiMessage().text();
            return parseVerdict(raw);
        } catch (Exception e) {
            log.error("[faithfulness] 判定调用失败：{}", e.toString());
            return new FaithfulnessVerdict(null, "判定调用失败：" + e.getMessage(), List.of());
        }
    }

    private String buildJudgePrompt(String question, List<String> sources, String answer) {
        StringBuilder references = new StringBuilder();
        for (int i = 0; i < sources.size(); i++) {
            String text = sources.get(i);
            if (text.length() > MAX_SOURCE_CHARS) {
                text = text.substring(0, MAX_SOURCE_CHARS) + "…";
            }
            references.append('[').append(i + 1).append("] ").append(text).append('\n');
        }
        return """
                你的任务是判断一段【回答】是否忠实于给定的【参考资料】。
                判据：回答里每一条事实性结论，都必须能在参考资料里找到依据。
                资料没有提到、但回答里说了的断言，算作"无依据"。
                回答只是复述或改写资料内容，则算忠实。
                只输出一个 JSON，不要任何解释文字、不要代码块标记，格式如下：
                {"faithful": true 或 false, "reason": "一句话说明理由", "unsupported": ["无依据的断言", "..."]}

                【参考资料】
                %s
                【问题】
                %s
                【回答】
                %s
                """.formatted(references, question, answer);
    }

    /** 解析判定结果。模型习惯把 JSON 包在代码块里，或者前后加一句话，所以按第一个 { 到最后一个 } 截取。 */
    static FaithfulnessVerdict parseVerdict(String raw) {
        if (raw == null || raw.isBlank()) {
            return new FaithfulnessVerdict(null, "判定返回为空", List.of());
        }
        int start = raw.indexOf('{');
        int end = raw.lastIndexOf('}');
        if (start < 0 || end <= start) {
            return new FaithfulnessVerdict(null, "判定返回不是 JSON：" + abbreviate(raw), List.of());
        }
        try {
            JsonNode root = MAPPER.readTree(raw.substring(start, end + 1));
            JsonNode faithful = root.path("faithful");
            if (!faithful.isBoolean()) {
                return new FaithfulnessVerdict(null, "判定结果缺少 faithful 字段", List.of());
            }
            List<String> unsupported = new ArrayList<>();
            for (JsonNode item : root.path("unsupported")) {
                unsupported.add(item.asText());
            }
            return new FaithfulnessVerdict(faithful.asBoolean(), root.path("reason").asText(""), unsupported);
        } catch (Exception e) {
            return new FaithfulnessVerdict(null, "判定 JSON 解析失败：" + e.getMessage(), List.of());
        }
    }

    private static String abbreviate(String text) {
        return text.length() <= 120 ? text : text.substring(0, 120) + "…";
    }

    /**
     * 引用检查结果。
     *
     * @param hasValidCitation 是否至少引用了一个存在的编号
     * @param citedIndexes     被引用到的编号（去重、升序）
     * @param invalidCitations 越界编号的个数
     */
    public record CitationCheck(boolean hasValidCitation, List<Integer> citedIndexes, int invalidCitations) {
    }

    /**
     * 忠实度判定结果。
     *
     * @param faithful            true/false 为判定结论；null 表示判定失败（单列计数，不算作忠实）
     * @param reason              判定理由
     * @param unsupportedClaims   判定出的无依据断言
     */
    public record FaithfulnessVerdict(Boolean faithful, String reason, List<String> unsupportedClaims) {
    }
}
