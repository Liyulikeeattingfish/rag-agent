package com.ragagent.rag;

/**
 * 写进每个 TextSegment 的元数据键名。
 *
 * 为什么要集中定义：这些字符串同时被「写入侧」（DocumentService）和「读取侧」（AskController）使用，
 * 散落成字面量的话，哪天改个名字就会出现"写入用 file_name、读取用 filename"这种静默失效的 bug——
 * 编译期完全发现不了，只会在运行时表现为"查不到来源"。
 */
public final class RagMetadataKeys {

    /**
     * 文档原始文件名。它就是本项目的「文档身份」：
     * 按文件名删除文档、按文件名判断"是不是同一份文档的新版本"。
     */
    public static final String FILE_NAME = "file_name";

    /** 整份文档内容的 SHA-256，用于内容级去重（同一份文档改了名字再传也认得出来）。 */
    public static final String CONTENT_HASH = "content_hash";

    /**
     * 该片段是这份文档切出来的第几块（从 0 开始）。
     * 注意：None 我们写的，是 LangChain4j 的切分器自动带上的，这里只是把键名登记下来方便读取。
     */
    public static final String SEGMENT_INDEX = "index";

    private RagMetadataKeys() {
    }
}
