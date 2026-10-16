package com.ragagent.controller;

import dev.langchain4j.model.chat.ChatModel;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * 第一天最小闭环：一个能"你问它答"的接口。
 * 启动后浏览器访问：http://localhost:8081/chat?message=你好
 */
@RestController
public class ChatController {

    private final ChatModel chatModel;

    // 构造器注入：Spring 会把 LLMConfig 里创建的 ChatModel 自动传进来
    public ChatController(ChatModel chatModel) {
        this.chatModel = chatModel;
    }

    @GetMapping("/chat")
    public String chat(@RequestParam("message") String message) {
        System.out.println("收到用户消息：" + message);
        // 把用户的话发给 DeepSeek，拿到并返回它的回答
        return chatModel.chat(message);
    }
}
