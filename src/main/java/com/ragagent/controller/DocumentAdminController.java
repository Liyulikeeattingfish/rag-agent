package com.ragagent.controller;

import com.ragagent.service.DocumentService;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 文档管理接口（入库侧的"运维面"）：看库里有哪些文档、按文件名整份删除。
 *
 * 用法：
 *   GET    /documents                    列出已入库文档（文件名 / 内容哈希 / 段数）
 *   DELETE /documents?filename=sample.txt 删除该文档的全部片段
 */
@RestController
@RequestMapping("/documents")
public class DocumentAdminController {

    private final DocumentService documentService;

    public DocumentAdminController(DocumentService documentService) {
        this.documentService = documentService;
    }

    @GetMapping
    public Map<String, Object> list() {
        return documentService.listDocuments();
    }

    @DeleteMapping
    public Map<String, Object> delete(@RequestParam("filename") String filename) {
        int removed = documentService.deleteByFileName(filename);

        Map<String, Object> result = new LinkedHashMap<>();
        // 设计选择：这里用 HTTP 200 + status 字段表达"这个文档不存在"，而不是返回 404。
        // 好处是调用方永远拿到同一个结构，解析逻辑简单；坏处是不够 RESTful。
        // 两种做法都行，关键是接口文档要写清楚，别让调用方猜。
        result.put("status", removed > 0 ? "deleted" : "not_found");
        result.put("filename", filename);
        result.put("removed", removed);
        return result;
    }
}
