package com.ragagent.service;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

/**
 * 忠实度检查的单测。
 *
 * 规则层是纯函数，必须钉死；模型层的价值在于"解析失败不能当成功" ——
 * 判定失败被算成忠实，等于把幻觉率刷成 0，这是最危险的一类错误。
 */
@ExtendWith(MockitoExtension.class)
class FaithfulnessCheckerTest {

    @Mock
    private ChatModel chatModel;

    private FaithfulnessChecker checker;

    @BeforeEach
    void setUp() {
        checker = new FaithfulnessChecker(chatModel);
    }

    @Test
    @DisplayName("规则层：识别有效引用、去重、并统计越界编号")
    void detectsValidAndInvalidCitations() {
        FaithfulnessChecker.CitationCheck check =
                checker.checkCitations("结论甲 [1]，结论乙 [3]，笔误 [9] 和 [0]", 3);

        assertThat(check.hasValidCitation()).isTrue();
        assertThat(check.citedIndexes()).containsExactly(1, 3);
        assertThat(check.invalidCitations()).isEqualTo(2);   // 9 和 0 都越界
    }

    @Test
    @DisplayName("规则层：一个引用都没标就是没标，不能算通过")
    void detectsMissingCitation() {
        FaithfulnessChecker.CitationCheck check = checker.checkCitations("这是一段没有任何引用的回答。", 3);

        assertThat(check.hasValidCitation()).isFalse();
        assertThat(check.citedIndexes()).isEmpty();
        assertThat(check.invalidCitations()).isZero();
    }

    @Test
    @DisplayName("规则层：没有资料时（拒答）不存在合法引用")
    void noSourcesMeansNoValidCitation() {
        assertThat(checker.checkCitations("资料中没有相关内容", 0).hasValidCitation()).isFalse();
    }

    @Test
    @DisplayName("模型层：解析带代码块围栏的 JSON，取出结论与无依据断言")
    void parsesVerdictWrappedInCodeFence() {
        when(chatModel.chat(any(ChatRequest.class))).thenReturn(ChatResponse.builder()
                .aiMessage(AiMessage.from("""
                        ```json
                        {"faithful": false, "reason": "第 2 句资料里没有", "unsupported": ["模型自己补的结论"]}
                        ```
                        """))
                .build());

        FaithfulnessChecker.FaithfulnessVerdict verdict =
                checker.judge("问题", List.of("资料一"), "回答 [1] 模型自己补的结论");

        assertThat(verdict.faithful()).isFalse();
        assertThat(verdict.reason()).contains("资料里没有");
        assertThat(verdict.unsupportedClaims()).containsExactly("模型自己补的结论");
    }

    @Test
    @DisplayName("模型层：返回不是 JSON 时标记为未判定，绝不能默认算作忠实")
    void unparsableVerdictIsMarkedUnknown() {
        when(chatModel.chat(any(ChatRequest.class))).thenReturn(ChatResponse.builder()
                .aiMessage(AiMessage.from("我觉得挺忠实的"))
                .build());

        FaithfulnessChecker.FaithfulnessVerdict verdict =
                checker.judge("问题", List.of("资料一"), "回答 [1]");

        assertThat(verdict.faithful()).isNull();
        assertThat(verdict.reason()).contains("不是 JSON");
    }

    @Test
    @DisplayName("模型层：调用抛异常也不能把异常冒泡打断整轮评测")
    void callFailureIsSwallowed() {
        when(chatModel.chat(any(ChatRequest.class))).thenThrow(new IllegalStateException("模型服务不可用"));

        FaithfulnessChecker.FaithfulnessVerdict verdict =
                checker.judge("问题", List.of("资料一"), "回答 [1]");

        assertThat(verdict.faithful()).isNull();
        assertThat(verdict.reason()).contains("判定调用失败");
    }
}
