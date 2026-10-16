package com.ragagent.controller;

import com.ragagent.service.DocumentService;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;

/**
 * 文档入库接口（RAG 的写入侧）。
 *
 * 这个类只做三件事：校验文件类型 → UTF-8 解码 → 转交给 DocumentService。
 * 查重规则、冲突判定、切块参数都属于业务规则，放在 Service 里。
 *
 * 用法：POST multipart/form-data，字段名 file；可选 overwrite=true（同名文档已存在且内容不同时覆盖旧版本）。
 */
@RestController
public class DocumentController {

    private final DocumentService documentService;

    public DocumentController(DocumentService documentService) {
        this.documentService = documentService;
    }

    @PostMapping(value = "/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Map<String, Object> upload(@RequestParam("file") MultipartFile file,
                                      @RequestParam(value = "overwrite", defaultValue = "false") boolean overwrite)
            throws IOException {
        String fileName = normalizeFileName(file.getOriginalFilename());

        // 只接受 .txt / .md，PDF 下一里程碑再做
        String lower = fileName.toLowerCase();
        if (!lower.endsWith(".txt") && !lower.endsWith(".md")) {
            throw new IllegalArgumentException("只支持 .txt / .md 文件，当前文件：" + fileName);
        }

        // UTF-8 读全文
        String text = new String(file.getBytes(), StandardCharsets.UTF_8);
        return documentService.ingest(fileName, text, overwrite);
    }

    /**
     * 只保留纯文件名，丢掉可能带上的目录部分。
     *
     * 为什么要做这一步：部分客户端（老的 IE、某些 SDK）会把完整路径当文件名传上来，
     * 比如 "C:\docs\a.txt"。如果原样存进 metadata，那么用户按 "a.txt" 去删除文档时就匹配不上，
     * 表现为"明明有这份文档却删不掉"。在入口处收敛成纯文件名，后面所有按文件名操作的逻辑才可靠。
     */
    private static String normalizeFileName(String original) {
        if (original == null || original.isBlank()) {
            throw new IllegalArgumentException("文件名不能为空");
        }
        String name = original.replace('\\', '/');
        int lastSlash = name.lastIndexOf('/');
        if (lastSlash >= 0) {
            name = name.substring(lastSlash + 1);
        }
        if (name.isBlank()) {
            throw new IllegalArgumentException("文件名不能为空：" + original);
        }
        return name;
    }
}
