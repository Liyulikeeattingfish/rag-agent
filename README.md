# rag-agent

基于 Spring Boot + LangChain4j 的 RAG 知识库问答服务，带 Agent 工具调用与可复现的检索/生成双层评测。

上传文档 → 切块向量化入库 → 提问时**混合召回 + RRF 融合 + 交叉编码精排** → 只把过闸的原文喂给大模型；
检索不到依据就直接拒答，而不是硬编。Agent 模式下由模型自己决定要不要查知识库。

它和"照着教程搭的 RAG"的区别只有一句：**检索质量和答案忠实度都是被评测和标定过的，不是靠肉眼看着还行。**
三份评测报告见 [docs/eval/EVAL-REPORT.md](docs/eval/EVAL-REPORT.md)。

## 技术栈

| 层 | 选型 |
|---|---|
| 应用 | Java 17 / Spring Boot 3.3.5 / LangChain4j 1.0.0 GA |
| 生成 | DeepSeek `deepseek-chat`（OpenAI 兼容协议） |
| 向量化 | 硅基流动 `Qwen/Qwen3-Embedding-0.6B`（1024 维） |
| 精排 | 硅基流动 `BAAI/bge-reranker-v2-m3`（cross-encoder） |
| 存储 | PostgreSQL 17 + pgvector（向量索引 HNSW）+ pg_trgm（关键词） |
| 测试 | JUnit 5 + Mockito + AssertJ（41 个单元测试） |

## 检索链路

~~~
用户问题
   │
   ├─→ 向量通道   EmbeddingStore.search(top10, minScore=0.66)   语义相近
   │
   └─→ 关键词通道  pg_trgm 或 BM25（top10）                      字面命中（专有名词）
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

两道闸分工不同，缺一不可：`min-score`（向量分）挡"语义上就不沾边"的问题；
`rerank-min-score` 挡"语义像但答非所问"的问题。**精排只排序、不拒答**，
所以拒答必须在它之后单独设一道闸 —— 这个结论在两套不同语料上各复现过一次。

为什么用 RRF 而不是把两路分数加权相加：两路分数都在 [0,1] 但**分布完全不同**
（不相关时向量分仍有 0.61，trigram 直接是 0），加权相加没有稳定区间；名次是无量纲的，
天然可比。代价是丢掉"第 1 名比第 2 名好多少"的信息，这个由精排补回来。

## Agent 工具调用（/agent）

`/ask` 是固定流程：检索一定会发生，模型没有选择权。`/agent` 把选择权交还给模型 ——
它自己决定要不要查知识库、要不要算数、还是直接回答。**有没有由模型做出的选择，就是 RAG 和 Agent 的分界线。**

暴露三个工具（`KnowledgeBaseTools`）：`searchKnowledgeBase`（把整条检索链路包成一个工具）、
`calculator`（四则运算，自己写的递归下降求值器）、`currentTime`。后两个本身没有技术含量，
它们的作用是**证明工具选择确实由模型做出**：问"现在几点"，模型知道该调 `currentTime` 而不是硬编一个时间。

循环是手写的（`AgentService`），没用 `AiServices` 注解那套，理由是可控制、可观测：

~~~
user → assistant(要求调用工具) → tool(结果) → assistant(最终回答)
~~~

四个必须自己处理的地方：

1. **assistant 那条消息必须回填进历史**。不回填的话，下一轮模型看不到自己要求过什么工具，
   OpenAI 兼容接口会直接报错 —— 这是手写循环最容易漏的一步。
2. **轮次上限**（`agent.max-rounds`，默认 5）。防的是模型反复调同一个工具：既烧 token，又让请求迟迟不返回。
3. **上限用尽不直接报错**，而是把工具摘掉再问一次，逼模型用已经拿到的信息收尾。
4. **工具异常一律转成文字回填**，不让它打断整轮对话；连"模型喊了一个不存在的工具"也回填可用工具清单。
   工具参数是不可信输入（可能为 null、可能不是合法 JSON），同样翻成文字回填。

## 关键词通道：pg_trgm 还是 BM25

同一个位置两个可替换实现，用 `--rag.retrieval.keyword-channel=trigram|bm25` 切换：

| | pg_trgm | BM25 |
|---|---|---|
| 打分 | 共享字符三元组比例 | IDF × 词频饱和 × 长度归一化 |
| 中文 | 免分词（三元组） | 需分词，这里用字符二元组近似 |
| 长处 | 零依赖、直接查库 | 排名更准（实测 Recall@1 +4.8 个点） |
| 短处 | 没有词频/IDF/长度归一化 | 召回过于宽松，**单独用会拉低拒答率** |

实测（同一套用例、其它配置完全一致）：BM25 的 Recall@1 92.86% vs pg_trgm 88.10%、MRR 0.9524 vs 0.9286；
但无关问题的拒答率从 60% 掉到 20% —— 它按"至少共享一个查询词"就返回，短查询几乎总能命中某个片段。
加上精排闸门之后两者收敛到完全一样：**关键词通道决定候选池，精排决定最终排序。**

## 向量索引

`rag_segments` 上建的是 **HNSW**（`m=16, ef_construction=64`，启动时自动创建），不是 IVFFlat：
IVFFlat 的簇划分依赖构建时的数据分布，表里没数据时建索引会得到退化的簇，HNSW 没有这个限制。

参数的选择不能靠背，得测。`GET /bench/vector-index` 在同一批数据上对比三种方案：

| 方案 | 服务端耗时 | 索引构建 | 索引体积 | Recall@10 | 执行计划 |
|---|:---:|:---:|:---:|:---:|---|
| 精确扫描（无索引） | 41.2 ms | — | — | 100% | Seq Scan |
| HNSW `ef_search=40` | **0.50 ms** | 11.4 s | 156 MB | 100% | Index Scan |
| HNSW `ef_search=100` | 0.62 ms | — | 156 MB | 100% | Index Scan |
| IVFFlat `lists=20, probes=4` | 3.23 ms | 0.8 s | 156 MB | 100% | Index Scan |

（2 万条 × 1024 维合成向量、200 次查询、top-10；执行计划是从返回值里带出来的，用来证明索引真的被用上了）

~~~bash
curl "http://localhost:8081/bench/vector-index?rows=20000&queries=200&topK=10&clusters=100"
~~~

三个必须说清楚的点：

- **合成数据不能用均匀随机**。1024 维里均匀随机的点几乎两两等距（维度灾难），ANN 召回会假性暴跌。
  这里用「聚类中心 + 高斯噪声」模拟真实 embedding 的邻域结构，且噪声要按 `σ/√D` 给，
  否则噪声范数会淹没中心向量（我第一版就是这么错的，测出来 recall 只有 15%）。
- **HNSW 构建本身有随机性**（节点层数是随机分配的），召回在多次运行间会在 98.5%~100% 浮动，
  这是它的固有性质，不是测量误差。
- **客户端耗时在这组实验里没有区分度**（四组 p50 都在 44~47 ms，服务端却差了 80 倍），
  所以以服务端 `EXPLAIN ANALYZE` 的 Execution Time 为准。这个现象记在报告里了，没查到底的事不做解释。

## 快速开始

前置：JDK 17、Maven、Docker Desktop。

~~~bash
# 1. 起向量库（PostgreSQL 17 + pgvector，宿主机端口 5433）
docker compose up -d

# 2. 配置密钥（真实密钥不能写在 application.yml 里，那个文件是要提交的）
cp src/main/resources/application-local.yml.example src/main/resources/application-local.yml
#    编辑 application-local.yml，填入 DeepSeek 与硅基流动的 api-key

# 3. 启动（端口 8081）
mvn spring-boot:run

# 4. 灌入评测语料（documents 目录下 11 份技术文档 + 项目根目录的 my-note.txt）
for f in docs/corpus/*.md; do curl -s -F "file=@$f" http://localhost:8081/upload; echo; done
curl -s -F "file=@my-note.txt" http://localhost:8081/upload
~~~

pg_trgm 扩展、GIN 索引与 HNSW 向量索引都由启动任务自动创建，不需要手工建。
`EmbeddingStore` 的建表也是启动时 `createTable(true)` 自动完成的。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/chat?message=` | 不检索，直连大模型（最小闭环，也用来验证密钥通不通） |
| POST | `/upload` | multipart 上传 `.txt`/`.md`；可选 `overwrite=true` 覆盖同名新版本 |
| GET | `/ask?question=` | RAG 问答（固定流程），返回 `answer` / `retrieved` / `sources` / `rerankUsed` |
| GET | `/agent?question=` | Agent 入口（模型自选工具），返回 `answer` / `toolCalls` / `rounds` |
| GET | `/documents` | 列出已入库文档（文件名 / 内容哈希 / 段数） |
| DELETE | `/documents?filename=` | 删除该文档的全部片段 |
| GET | `/eval` | 检索侧评测：47 条用例，输出 Recall@3/@1、MRR、nDCG@3、拒答率。**不调大模型** |
| GET | `/eval-answer` | 生成侧评测：引用覆盖率 + 忠实率。要调大模型，可加 `?limit=10` 冒烟 |
| GET | `/bench/vector-index` | 向量索引压测：精确扫描 / HNSW / IVFFlat 的召回与延迟对比 |

`/ask` 的返回体结构**只增不删**：`score` 是老字段（等于向量分），新增 `vectorScore` /
`keywordScore` / `rrfScore` / `rerankScore`。某一路没参与就是 `null` 而不是 0 ——
`null` 表示"没参与"，0 表示"算出来是 0"。

## 评测结果

### 检索侧：四档消融 + 关键词通道 A/B

评测集 47 条（42 条有答案 + 5 条无关），语料 12 份文档 22 个片段。**只改配置开关，不改一行代码。**

| 档 | 关键词通道 | hybrid | rerank | rerankMinScore | Recall@3 | Recall@1 | MRR | nDCG@3 | 拒答率 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 纯向量基线 | — | ✗ | ✗ | — | 95.24% | 85.71% | 0.9048 | 0.9172 | 60% |
| 2 混合召回 + RRF | pg_trgm | ✓ | ✗ | — | 97.62% | 88.10% | 0.9286 | 0.9410 | 60% |
| 3 + 精排（阈值占位 0.0） | pg_trgm | ✓ | ✓ | 0.0 | 100% | 97.62% | 0.9881 | 0.9912 | 60% |
| 4 + 精排拒答闸＝默认配置 | pg_trgm | ✓ | ✓ | 0.05 | 100% | 97.62% | 0.9881 | 0.9912 | **100%** |
| 2b 同档 2，换 BM25 | **BM25** | ✓ | ✗ | — | 97.62% | **92.86%** | **0.9524** | **0.9586** | 20% |
| 4b 同档 4，换 BM25 | **BM25** | ✓ | ✓ | 0.05 | 100% | 97.62% | 0.9881 | 0.9912 | 100% |

四个结论：

1. **混合召回**把 Recall@3 从 95.24% 提到 97.62%、Recall@1 从 85.71% 提到 88.10%。
2. **精排**把 Recall@1 从 88.10% 提到 97.62%、MRR 从 0.9286 提到 0.9881 ——
   它解决的不是"能不能捞回来"，而是"捞回来的排在第几"。
3. **拒答只能靠精排之后的阈值闸**：档 1~3 的拒答率都是 60%，只有档 4 把 `rerank-min-score`
   设成 0.05 才到 100%。换了一套语料，这个结论重演了一次。
4. **BM25 排名更准但拒答更差**（见上一节）—— 更好的关键词通道不一定让整条链路更好，
   它把压力全推给了后面的闸门。

### 生成侧：引用溯源 + 忠实度

42 条正例跑完整回答链路，度量三个不同的失败：

| 指标 | 结果 | 含义 |
|---|:---:|---|
| 漏答率 | **0%** | 正例被检索层拒答的比例（检索没召回） |
| 引用覆盖率 | **97.62%** | 回答里标了有效引用编号 `[1]` 的比例（规则层，零成本） |
| 忠实率 | **97.62%** | 判定为"每句都有资料依据"的比例（LLM-as-judge） |
| 越界引用 | 0 条 | 引用了不存在的编号 |

唯一一条失败很有意思，也值得单独说：**它是"该答却拒答"，不是幻觉。**
问题是"重排模型会不会自己拒绝回答不相关的问题"，检索回来的资料里明确写了"重排只排序不拒答"，
模型却回了"资料中没有相关内容"。它既没标引用（所以把引用覆盖率拉低），也被判为不忠实。
这说明**忠实度这个指标单看会把"过度保守"也算进去**，必须和引用覆盖率、漏答率一起看。

为什么和 `/eval` 分成两个接口：`/eval` 不调大模型，秒级可复现，每次改代码都能跑；
`/eval-answer` 要调两次大模型（回答 + 判定），分钟级、有随机性、成本高，只能定期跑。
把两者混在一起，快的被慢的拖死，可复现的变得不可复现。

## 测试

~~~bash
mvn test
~~~

41 个单元测试，覆盖那些**改坏了不会报错、只会悄悄失灵**的地方：

- `RetrievalServiceTest`（8）：RRF 融合与去重身份、精排阈值过滤、**精排异常降级**
  （不能拿 null 分数去过滤，否则候选会被整批清空）、纯向量基线路径。
- `AgentServiceTest`（5）：**消息回填顺序**（少了 assistant 那条接口会报错）、轮次上限与强制收尾、
  工具异常兜底、未知工具兜底。
- `DocumentServiceTest`（4）：内容级去重、同名冲突默认拒绝、覆盖时先删后写、缓存失效时机。
- `Bm25KeywordSearcherTest`（6）+ `KnowledgeBaseToolsTest`（5）+ `ArithmeticEvaluatorTest`（3）
  + `RetrievalMetricsTest`（4）+ `FaithfulnessCheckerTest`（6）：分词与打分、参数解析、求值器边界、
  指标口径、判定失败的降级。

`RagAgentApplicationTests`（整体容器启动）标了 `@Disabled`：它需要本地 pgvector，不应该拖累 `mvn test`。

> 测试用的是 Mockito 的 **subclass mock maker**（`src/test/resources/mockito-extensions/`），
> 不走 ByteBuddy 的 agent 自挂载，因此在受限容器 / CI 里也能跑起来。

## 目录结构

~~~
src/main/java/com/ragagent/
├── config/
│   ├── LLMConfig.java             ChatModel（DeepSeek）
│   ├── EmbeddingConfig.java       EmbeddingModel + PgVectorEmbeddingStore
│   ├── RetrievalProperties.java   检索全参数，prefix=rag.retrieval，字段自带默认值
│   ├── PgTrgmInitializer.java     启动时建 pg_trgm 扩展与 GIN 索引
│   └── VectorIndexInitializer.java 启动时建 HNSW 向量索引
├── controller/                    Chat / Document / DocumentAdmin / Ask / Agent / Eval / AnswerEval / Bench
├── service/
│   ├── DocumentService.java       入库幂等、按文档删除、列出文档
│   ├── RetrievalService.java      混合召回 → RRF → 精排 → 双闸门
│   ├── AnswerService.java         检索 → 拼 prompt → 生成（/ask 与生成侧评测共用）
│   ├── AgentService.java          Agent 主循环：工具调用 → 回填 → 再问，含轮次上限
│   ├── FaithfulnessChecker.java   引用规则校验 + LLM-as-judge 忠实度判定
│   ├── AnswerEvaluationService.java 生成侧评测编排
│   └── RerankClient.java          硅基流动 rerank 客户端
├── agent/                         AgentTool / AgentResult / KnowledgeBaseTools / ArithmeticEvaluator
├── rag/                           TrigramKeywordSearcher / Bm25KeywordSearcher / KeywordChannel /
│                                  KeywordHit / RetrievedSegment / RetrievalMetrics / RagMetadataKeys
├── eval/                          EvalCase / EvalCaseSet（检索侧与生成侧共用同一套用例）
└── bench/                         VectorIndexBenchmark（精确扫描 / HNSW / IVFFlat 对比）
~~~

入库侧的一处设计值得单独说：`file_name` 是文档身份，`content_hash` 是整份文档的 SHA-256。
查重先看内容（改个名字重传也认得出来），再看文件名（同名但内容不同 = 这份文档的新版本）。
同名冲突**默认拒绝**而不是自动覆盖 —— 新旧片段并存会让检索同时召回旧答案，直接污染结果。

## 路线图

- [x] M0 工程骨架
- [x] M1 最小闭环（`/chat` 直连 DeepSeek）
- [x] M2 RAG（`/upload` 切块入库、`/ask` 检索问答、`/documents` 管理）
- [x] M3 检索质量增强（混合召回 + RRF + 精排 + 双闸门 + 检索侧评测）
- [x] M4 Agent 工具调用（ToolSpecification，手写 tool-execution 循环）
- [x] M5 向量索引与规模压测（HNSW / IVFFlat 对比）
- [x] M6 引用溯源与忠实度评测（生成侧）
- [ ] M7 多轮对话（会话历史 + 指代消解式 query 改写）
- [ ] M8 PDF 解析、异步入库、SSE 流式输出、部署

## 已知限制

- 检索评测语料 12 份 22 段，仍然偏小；**指标的绝对值不宜外推**，有意义的是档位之间的相对变化。
  规模相关的结论（索引、延迟）另用 2 万条合成向量单独测。
- 关键词通道的 GIN 索引**不在 pg_trgm 查询的关键路径上**：查询用的是 `similarity(text,?)`（先算分再比较），
  走顺序扫描；GIN 加速的是 `%` 运算符。语料变大后应改成 `WHERE text % ?` 并 `SET LOCAL pg_trgm.similarity_threshold`。
- BM25 用的是内存倒排索引，语料变更后靠 `DocumentService` 调 `invalidate()` 作废；
  如果有人绕过接口直接写库，缓存不会失效。
- 切块 500/100 是经验起点，**没有做过切块参数消融**。
- 入库是同步的：大文档会在 `embedAll` 上阻塞请求。
- 无鉴权、无多租户隔离，`/ask` 用的是 GET。
