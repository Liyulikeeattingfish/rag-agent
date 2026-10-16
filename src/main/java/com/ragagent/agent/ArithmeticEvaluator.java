package com.ragagent.agent;

/**
 * 极小的四则运算求值器（递归下降），供 calculator 工具使用。
 *
 * 为什么不直接把表达式丢给脚本引擎：一是 JDK 15 起 Nashorn 已被移除，JDK 17 上没有开箱可用的
 * JS 引擎；二是把整段字符串交给脚本引擎求值，等于开放了一个任意代码执行面，
 * 而模型生成的参数是**不可信输入**。在能力上做减法（只认 + - * / 和括号）是刻意的选择。
 *
 * 文法（一层一个方法，这也是递归下降最容易讲清楚的地方）：
 *   expression := term (('+' | '-') term)*
 *   term       := factor (('*' | '/') factor)*
 *   factor     := ('+' | '-') factor | number | '(' expression ')'
 */
public final class ArithmeticEvaluator {

    /** 表达式长度上限：防呆，也防止有人拿这个工具当免费计算器刷。 */
    private static final int MAX_LENGTH = 200;

    private final String input;
    private int position;

    private ArithmeticEvaluator(String input) {
        this.input = input;
    }

    public static double evaluate(String expression) {
        if (expression == null || expression.isBlank()) {
            throw new IllegalArgumentException("表达式为空");
        }
        // 顺手接受空格与全角乘除号，模型偶尔会输出这些
        String normalized = expression.replace(" ", "").replace("×", "*").replace("÷", "/");
        if (normalized.length() > MAX_LENGTH) {
            throw new IllegalArgumentException("表达式过长（上限 " + MAX_LENGTH + " 个字符）");
        }
        ArithmeticEvaluator evaluator = new ArithmeticEvaluator(normalized);
        double value = evaluator.parseExpression();
        if (evaluator.position != normalized.length()) {
            throw new IllegalArgumentException("存在无法识别的字符：" + normalized.charAt(evaluator.position));
        }
        return value;
    }

    private double parseExpression() {
        double value = parseTerm();
        while (position < input.length()) {
            char c = input.charAt(position);
            if (c == '+') {
                position++;
                value += parseTerm();
            } else if (c == '-') {
                position++;
                value -= parseTerm();
            } else {
                break;
            }
        }
        return value;
    }

    private double parseTerm() {
        double value = parseFactor();
        while (position < input.length()) {
            char c = input.charAt(position);
            if (c == '*') {
                position++;
                value *= parseFactor();
            } else if (c == '/') {
                position++;
                double divisor = parseFactor();
                if (divisor == 0) {
                    throw new ArithmeticException("除数不能为 0");
                }
                value /= divisor;
            } else {
                break;
            }
        }
        return value;
    }

    private double parseFactor() {
        if (position >= input.length()) {
            throw new IllegalArgumentException("表达式不完整");
        }
        char c = input.charAt(position);
        if (c == '+') {
            position++;
            return parseFactor();
        }
        if (c == '-') {
            position++;
            return -parseFactor();
        }
        if (c == '(') {
            position++;
            double value = parseExpression();
            if (position >= input.length() || input.charAt(position) != ')') {
                throw new IllegalArgumentException("括号不匹配");
            }
            position++;
            return value;
        }
        return parseNumber();
    }

    private double parseNumber() {
        int start = position;
        boolean dotSeen = false;
        while (position < input.length()) {
            char c = input.charAt(position);
            if (Character.isDigit(c)) {
                position++;
            } else if (c == '.' && !dotSeen) {
                dotSeen = true;
                position++;
            } else {
                break;
            }
        }
        if (start == position) {
            throw new IllegalArgumentException("这里应该是一个数字，实际是：" + input.charAt(position));
        }
        return Double.parseDouble(input.substring(start, position));
    }
}
