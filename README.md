# rag-agent

基于 Spring Boot + LangChain4j 的 RAG 知识库问答服务。

上传文档 → 切块向量化入库 → 提问时**混合召回 + RRF 融合 + 交叉编码精排** → 只把过闸的原文喂给大模型；
检索不到够格资料就直接拒答，而不是硬编。

它和"照着教程搭的 RAG"的区别只有一句：**检索质量是被评测和标定过的，不是靠肉眼看着还行。**
四档消融的实测报告见 [docs/eval/RETRIEVAL-EVAL-REPORT.md](docs/eval/RETRIEVAL-EVAL-REPORT.md)。

## 技术栈

| 层 | 选型 |
|---|---|
| 应用 | Java 17 / Spring Boot 3.3.5 / LangChain4j 1.0.0 GA |
| 生成 | DeepSeek `deepseek-chat`（OpenAI 兼容协议） |
| 向量化 | 硅基流动 `Qwen/Qwen3-Embedding-0.6B`（1024 维） |
| 精排 | 硅基流动 `BAAI/bge-reranker-v2-m3`（cross-encoder） |
| 存储 | PostgreSQL 17 + pgvector（向量）+ pg_trgm（关键词） |
| 测试 | JUnit 5 + Mockito + AssertJ |

## 检索链路

~~~
用户问题
   │
   ├─→ 向量通道   EmbeddingStore.search(top10, minScore=0.66)   语义相近
   │
   └─→ 关键词通道  pg_trgm similarity(top10)                     字面命中（专有名词）
              │
              ▼
        RRF 融合去重（只取名次：Σ 1/(k+rank)，k=60）→ 候选 10 条
              │
              ▼
        bge-reranker 交叉编码精排（query 与候选逐条对比）
              │
              ▼
        过 rerank-min-score=0.05 闸门 → 取 topN=3 → 拼 prompt → DeepSeek
              │
        （任一道闸全灭 → 直接返回"资料中没有相关内容"，不调用大模型）
~~~

两道闸分工不同，缺一不可：

- `min-score`（向量分）挡"语义上就不沾边"的问题；
- `rerank-min-score` 挡"语义像但答非所问"的问题。

为什么用 RRF 而不是把两路分数加权相加：两路分数都在 [0,1] 但**分布完全不同**
（不相关时向量分仍有 0.61，trigram 直接是 0），加权相加没有稳定区间；名次是无量纲的，
天然可比。代价是丢掉"第 1 名比第 2 名好多少"的信息，这个由精排补回来。

## 快速开始

前置：JDK 17、Maven、Docker Desktop。

~~~bash
# 1. 起向量库（PostgreSQL 17 + pgvector，宿主机端口 5433）
docker compose up -d

# 2. 配置密钥
#    真实密钥不能写在 application.yml 里（那个文件是要提交的）
cp src/main/resources/application-local.yml.example src/main/resources/application-local.yml
#    编辑 application-local.yml，填入 DeepSeek 与硅基流动的 api-key
#    也可以用环境变量 DEEPSEEK_API_KEY / SILICONFLOW_API_KEY 覆盖

# 3. 启动（端口 8081）
mvn spring-boot:run
~~~

pg_trgm 扩展与 GIN 索引由 `PgTrgmInitializer` 在启动时自动创建，不需要手工建。
`EmbeddingStore` 的建表也是启动时 `createTable(true)` 自动完成的。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/chat?message=` | 不检索，直连大模型（最小闭环，也用来验证密钥通不通） |
| POST | `/upload` | multipart 上传 `.txt`/`.md`；可选 `overwrite=true` 覆盖同名新版本 |
| GET | `/ask?question=` | RAG 问答，返回 `answer` / `retrieved` / `sources` / `rerankUsed` |
| GET | `/documents` | 列出已入库文档（文件名 / 内容哈希 / 段数） |
| DELETE | `/documents?filename=` | 删除该文档的全部片段 |
| GET | `/eval` | 跑 20 条用例的检索评测，返回四指标 + 逐条明细 + 生效配置回显 |

`/ask` 的返回体结构**只增不删**：`score` 是老字段（等于向量分），新增 `vectorScore` /
`keywordScore` / `rrfScore` / `rerankScore`。某一路没参与就是 `null` 而不是 0 ——
`null` 表示"没参与"，0 表示"算出来是 0"。

~~~bash
curl "http://localhost:8081/ask?question=vector_cosine_ops"
curl -F "file=@notes.md" "http://localhost:8081/upload"
curl "http://localhost:8081/eval"
~~~

## 检索质量：四档消融

评测集 20 条（16 条有答案 + 4 条无关问题），语料 3 份文档 5 个片段。
四档之间**只改配置开关**，用启动参数覆盖，不改一行代码：

| 档 | hybrid | rerank | rerankMinScore | Recall@3 | Recall@1 | MRR | 拒答率 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 纯向量基线 | ✗ | ✗ | — | 93.8% | 75.0% | 0.8438 | 75.0% |
| 2 混合召回 + RRF | ✓ | ✗ | — | 100% | 87.5% | 0.9375 | 50.0% |
| 3 + 精排（阈值占位 0.0） | ✓ | ✓ | 0.0 | 100% | 100% | 1.0 | 50.0% |
| 4 + 精排拒答闸（标定后 = 默认配置） | ✓ | ✓ | 0.05 | 100% | 100% | 1.0 | 100% |

复现（档 1 为例，其余同理）：

~~~bash
mvn spring-boot:run -Dspring-boot.run.arguments="--rag.retrieval.hybrid-enabled=false --rag.retrieval.rerank-enabled=false"
curl http://localhost:8081/eval
~~~

三个值得记住的结论：

1. **加了混合检索之后，拒答率反而掉了一半**（档 2、档 3 都是 50%，低于档 1 的 75%）。
   原因是候选池从"向量 top3"扩大到"10 条融合候选"，噪声挤进前 3 的机会变多；
   而 rerank **只排序、不拒答**，把不相关片段压到 0.001 也照样会返回。
2. **所以必须再补一道闸。** 实测分数分布：相关用例 rank1 落在 0.17~0.9993，
   无关用例最高只有 0.0014 —— 中间是约 100 倍宽的空档，`0.05` 就取在里面。
   设上之后拒答率 50% → 100%，而正例指标完全不受影响。
3. **关键词通道救回的是"专有名词"。** 基线档下「精灵小窝是什么」检索结果为空
   （embedding 表达不了自己笔记里的专有名词），混合档排到第 1。

> ⚠️ 语料只有 3 份 5 段，**指标的绝对值没有外推意义**。有价值的是四档之间的相对变化和标定
> 方法本身；另外小语料下 Recall@3 必然饱和，所以同时输出 Recall@1 与 MRR 才有分辨率。

## 测试

~~~bash
mvn test
~~~

12 个单元测试，覆盖三处最容易改坏、而且**改坏了不会报错**的地方：

- `RetrievalServiceTest`（8 个）：RRF 融合与去重身份、精排阈值过滤、
  **精排服务异常时的降级**（不能拿 null 分数去过滤，否则候选会被整批清空）、
  纯向量基线路径、finalTopK 截断。
- `DocumentServiceTest`（4 个）：内容级去重、同名冲突默认拒绝、`overwrite=true` 先删后写、全新文档入库。

`RagAgentApplicationTests`（整体容器启动）标了 `@Disabled`：它需要本地 pgvector，
不应该拖累 `mvn test`，要跑就先去 `docker compose up -d` 再摘掉注解。

> 测试用的是 Mockito 的 **subclass mock maker**（`src/test/resources/mockito-extensions/`），
> 不走 ByteBuddy 的 agent 自挂载，因此在受限容器 / CI 里也能跑起来。

## 目录结构

~~~
src/main/java/com/ragagent/
├── config/
│   ├── LLMConfig.java             ChatModel（DeepSeek）
│   ├── EmbeddingConfig.java       EmbeddingModel + PgVectorEmbeddingStore
│   ├── RetrievalProperties.java   检索全参数，prefix=rag.retrieval，字段自带默认值
│   └── PgTrgmInitializer.java     启动时建 pg_trgm 扩展与 GIN 索引
├── controller/
│   ├── ChatController.java        最小闭环
│   ├── DocumentController.java    /upload（只做协议与校验）
│   ├── DocumentAdminController.java /documents 列表与删除
│   ├── AskController.java         /ask（只做调检索 + 拼 prompt）
│   └── EvalController.java        /eval 评测用例与指标计算
├── service/
│   ├── DocumentService.java       入库幂等、按文档删除、列出文档
│   ├── RetrievalService.java      混合召回 → RRF → 精排 → 双闸门
│   └── RerankClient.java          硅基流动 rerank 客户端
└── rag/
    ├── KeywordSearcher.java       pg_trgm 关键词通道
    ├── RetrievedSegment.java      四路分数（vector/keyword/rrf/rerank）
    └── RagMetadataKeys.java       元数据键常量，写入侧读取侧共用
~~~

入库侧的一处设计值得单独说：`file_name` 是文档身份，`content_hash` 是整份文档的 SHA-256。
查重先看内容（改个名字重传也认得出来），再看文件名（同名但内容不同 = 这份文档的新版本）。
同名冲突**默认拒绝**而不是自动覆盖 —— 新旧片段并存会让检索同时召回旧答案，直接污染结果；
覆盖是破坏性操作，必须由调用方显式传 `overwrite=true`。

## 路线图

- [x] M0 工程骨架
- [x] M1 最小闭环（`/chat` 直连 DeepSeek）
- [x] M2 RAG（`/upload` 切块入库、`/ask` 检索问答、`/documents` 管理）
- [x] M2.5 检索质量增强（混合召回 + RRF + 精排 + 双闸门拒答 + `/eval` 消融评测）
- [ ] M3 多轮对话（会话历史 + 指代消解式 query 改写）
- [ ] M4 Agent 工具调用（ToolSpecification / Function Calling）
- [ ] M5 答案引用溯源 + 忠实度校验
- [ ] M6 PDF 解析、异步入库、向量索引（HNSW）、SSE 流式输出、部署

## 已知限制

- 语料规模小（3 份 5 段），评测指标的绝对值不可外推；扩语料 + 做规模测试是下一步。
- 关键词通道的 GIN 索引**不在当前查询的关键路径上**：查询用的是 `similarity(text,?)`（先算分再比较），
  走顺序扫描；GIN 加速的是 `%` 运算符。语料变大后应改成 `WHERE text % ? ORDER BY similarity(...)`
  并 `SET LOCAL pg_trgm.similarity_threshold = 0`。索引已按计划建好，作为这条演进路径的起点。
- 没有 HNSW / IVFFlat 索引，向量检索目前是精确扫描。
- 切块 500/100 是经验起点，**没有做过切块参数消融**。
- 入库是同步的：大文档会在 `embedAll` 上阻塞请求。
- 无鉴权、无多租户隔离，`/ask` 用的是 GET。
