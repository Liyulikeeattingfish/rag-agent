package com.ragagent.agent;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** calculator 工具的求值逻辑 —— 纯函数，最该被钉死的地方。 */
class ArithmeticEvaluatorTest {

    @Test
    @DisplayName("优先级、括号、负号、小数、全角乘号都能算对")
    void evaluatesValidExpressions() {
        assertThat(ArithmeticEvaluator.evaluate("1+2*3")).isEqualTo(7.0);
        assertThat(ArithmeticEvaluator.evaluate("(1+2)*3")).isEqualTo(9.0);
        assertThat(ArithmeticEvaluator.evaluate("10/4")).isEqualTo(2.5);
        assertThat(ArithmeticEvaluator.evaluate("-3+5")).isEqualTo(2.0);
        assertThat(ArithmeticEvaluator.evaluate("2*(3+(4-1))")).isEqualTo(12.0);
        assertThat(ArithmeticEvaluator.evaluate("1.5*2")).isEqualTo(3.0);
        assertThat(ArithmeticEvaluator.evaluate("12 × 2")).isEqualTo(24.0);
    }

    @Test
    @DisplayName("非法表达式一律抛异常，由上层转成文字回填给模型让它自己纠错")
    void rejectsMalformedExpressions() {
        assertThatThrownBy(() -> ArithmeticEvaluator.evaluate("1/0")).isInstanceOf(ArithmeticException.class);
        assertThatThrownBy(() -> ArithmeticEvaluator.evaluate("(1+2")).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> ArithmeticEvaluator.evaluate("1+")).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> ArithmeticEvaluator.evaluate("1+a")).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> ArithmeticEvaluator.evaluate("   ")).isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    @DisplayName("表达式长度有上限：模型生成的参数不可信，不能让这个工具变成免费计算器")
    void rejectsOverlyLongExpression() {
        String longExpression = "1+".repeat(200) + "1";
        assertThatThrownBy(() -> ArithmeticEvaluator.evaluate(longExpression))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("过长");
    }
}
