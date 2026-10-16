> ⚠️ **这是上一版报告（3 份文档 / 5 个片段 / 20 条用例）。语料扩充之后的完整评测见同目录的
> [EVAL-REPORT.md](EVAL-REPORT.md)** —— 那份才是当前数字。本文件保留下来，
> 是因为它是"同一结论在不同语料上复现"的证据。

# 检索质量增强 — 消融评测报告

评测接口：`GET /eval`（20 条用例：16 条正例 + 4 条无关用例，不调用大模型）
语料：`my-note.txt`(1段) / `sample.txt`(2段) / `vector-db-notes-v2.md`(2段)，共 3 份 5 段
环境：pgvector(5433) + Qwen3-Embedding-0.6B(1024维) + BAAI/bge-reranker-v2-m3

## 一、四档对比

| # | 档位 | hybrid | rerank | rerankMinScore | Recall@3 | Recall@1 | MRR | 拒答率 |
|---|------|:------:|:------:|:--------------:|:--------:|:--------:|:---:|:------:|
| 1 | 改造前基线（纯向量） | ✗ | ✗ | — | 93.8% | **75.0%** | 0.8438 | 75.0% |
| 2 | 混合召回 + RRF | ✓ | ✗ | — | 100% | **87.5%** | 0.9375 | 50.0% |
| 3 | + rerank 精排（阈值占位 0.0） | ✓ | ✓ | 0.0 | 100% | **100%** | **1.0** | 50.0% |
| 4 | + rerank 拒答闸（标定后） | ✓ | ✓ | 0.05 | 100% | **100%** | **1.0** | **100%** |

三档消融（1→2→3）是配置开关跑出来的真值；第 4 档是在第 3 档基础上只改一个阈值。
所有档位配置均通过启动参数覆盖，未改动 `application.yml`：

```bash
# 档 1
--rag.retrieval.hybrid-enabled=false --rag.retrieval.rerank-enabled=false
# 档 2
--rag.retrieval.hybrid-enabled=true  --rag.retrieval.rerank-enabled=false
# 档 3（默认开关 + 阈值归零）
--rag.retrieval.rerank-min-score=0.0
# 档 4 = 纯默认配置，什么都不加
```

## 二、每一档到底修好了什么

### 档 1 → 档 2：关键词通道补上「字面命中」的短板

基线档 4 条没排到第一位，混合召回后剩 2 条：

| 用例 | 期望文档 | 基线 | 混合 |
|------|---------|:----:|:----:|
| 向量数据库和关系数据库有什么区别 | vector-db-notes-v2.md | rank 2 | rank 1 |
| 精灵小窝是什么 | my-note.txt | **检索不到** | rank 1 |
| vector_cosine_ops 是什么 | vector-db-notes-v2.md | rank 2 | rank 2 |
| pgvector 的 probes 参数是干什么的 | vector-db-notes-v2.md | rank 2 | rank 2 |

最典型的是 **「精灵小窝是什么」在基线档下检索结果为空**。这是自己笔记里的专有名词，
embedding 无法表达，纯向量彻底翻车；关键词通道靠 trigram 字面交集把它捞了回来。
但另外两条只靠 RRF 仍到不了第 1 名 —— RRF 只负责"把对的捞进候选池"，排不好名次。

### 档 2 → 档 3：rerank 解决「排好名次」

`vector_cosine_ops`、`probes` 这两条从 rank 2 提到 rank 1，16 条正例全部命中首位。
rerank 拿 query 和候选逐条做交叉编码，比"两路名次求和"的 RRF 精确得多。

### 档 3 → 档 4：必须补一道拒答闸

**值得注意的是档 3 的拒答率反而从 75% 掉到 50%** —— 这不是 bug，是混合检索的必然副作用：
候选池从"向量 top3"扩大到"10 条融合候选"，噪声挤进前 3 的机会变多了。
而 rerank **只排序、不拒答**，它把不相关片段的分压到 0.001 也照样会返回，除非设阈值。

实测分数分布（这正是 `rerankMinScore = 0.05` 的标定依据）：

| 类别 | rerank 分区间 |
|------|--------------|
| 相关用例 rank1 | 0.17 ~ 0.9993 |
| 无关用例（今天天气 / 推荐电影 / 炒西红柿鸡蛋 / Python 的 GIL） | 0.0002 ~ 0.0014 |

中间是一条约 100 倍宽的空档，0.05 落在里面，两端各有 35 倍 / 3.5 倍余量。
设上之后拒答率 50% → 100%，且正例指标完全不受影响。

## 三、两个必须记住的结论

1. **两道闸拦的是两类不同的东西，缺一不可。**
   - 粗召回 `min-score`（向量分闸）挡"语义上就不沾边"的 —— 今天天气、炒西红柿鸡蛋。
   - `rerank-min-score` 挡"语义像但答非所问"的 —— 「Python 的 GIL 是什么」语料里也有
     Python/语言字眼，向量分能过 0.66 的闸，只有 rerank 能把它压到 0.0014。
2. **粗召回必须留闸。** 若把 `min-score` 放开，任何问题都会稳定返回 10 条候选，
   后面的 rerank 只排序不拒答，闸一失守就再也堵不上了。

## 四、复现

```sql
-- 语料指纹（用于确认是同一套评测数据）
SELECT metadata->>'file_name', metadata->>'index', length(text) FROM rag_segments ORDER BY 1,2;
```

```
GET  /eval                      # 四指标 + 逐条明细 + 生效配置回显
GET  /ask?question=xxx          # sources 带 vectorScore/keywordScore/rrfScore/rerankScore
GET  /documents
POST /upload                    # multipart file
DELETE /documents?filename=xxx
```

原始结果 JSON：`baseline-vector-only.json` / `hybrid-no-rerank.json` /
`full-pipeline.json`（阈值 0.0）/ `full-pipeline-threshold.json` / `default-config.json`（同目录）

---

## 五、复现验证（2026-10-16 重跑）

上面四档的每一个数字都在本机重跑过一遍，开关全部通过启动参数覆盖、未改代码；
并且从 `/eval` 返回体的 `config` 字段确认了每一档**真的**跑在预期配置下（不是"以为改了、其实没改"）：

| 档 | 生效配置（`/eval` 回显） | Recall@3 | Recall@1 | MRR | 拒答率 |
|---|---|---|---|---|---|
| 1 | hybrid=false rerank=false rerankMinScore=0.05 | 93.8% | 75.0% | 0.8438 | 75.0% |
| 2 | hybrid=true rerank=false rerankMinScore=0.05 | 100% | 87.5% | 0.9375 | 50.0% |
| 3 | hybrid=true rerank=true rerankMinScore=0.0 | 100% | 100% | 1.0 | 50.0% |
| 4 | hybrid=true rerank=true rerankMinScore=0.05 | 100% | 100% | 1.0 | 100.0% |

重跑的原始结果 JSON 在 `docs/eval/rerun/`（档 1~4 各一份）。

### 复现时踩到的一个坑（值得记下来）

第一次重跑四档时，**四个档位的结果一模一样、全都是默认配置的数值**。
根因：重启脚本用 `Get-NetTCPConnection -LocalPort 8081` 找进程来 kill，而它在这个环境里静默返回空，
于是旧实例根本没被杀掉；新实例因为 8081 被占用直接启动失败退出，`/eval` 打到的始终是那个还在跑的旧实例。
改成**按端口另起实例（8082~8085）之后再关闭**，才拿到真值。

教训：做消融对比时"配置到底有没有生效"必须从返回体里回显确认，不能只看命令行里写了什么 ——
这正是 `/eval` 要把 `config` 一起返回的原因。
