# RAG 评测指标

## 检索侧指标
- Recall@k 与 Hit@k：期望文档是否出现在前 k 个结果里。只有一个相关文档时两者等价，衡量"能不能捞回来"。
- MRR（Mean Reciprocal Rank）：正确文档排名的倒数的均值，没进前 k 记 0。比 Recall 更有分辨率，
  它能看出"从第 3 名提到第 1 名"这种进步。
- nDCG@k（归一化折损累计增益）：DCG 等于 Σ (2^rel 减 1) / log2(i 加 1)，IDCG 是理想排序的 DCG，
  nDCG 等于 DCG 除以 IDCG。它同时考虑相关性和位置，有分级相关性标注时也能用。
  只有 0/1 标注时，单相关文档下 nDCG@k 退化成 1 / log2(rank 加 1)。

## 生成侧指标
- Faithfulness 与 Groundedness：答案里的每一句是否都能被检索到的资料支撑，衡量幻觉。
  常用做法是 LLM-as-judge：把资料和答案一起给模型，问"有没有资料里没有的断言"。
- Answer Relevancy：答案是否切题，和问题意图是否一致。
- Context Recall 与 Context Precision：检索到的上下文里有多少是真正需要的、有多少是噪声。

## 体系设计上的取舍
把 LLM 掺进评测会让结果不可重复、又慢又贵。所以检索侧和生成侧应该分成两套：
检索侧不调 LLM（秒级、可复现、每次改代码都能跑），生成侧调 LLM（分钟级、有随机性、定期跑）。

