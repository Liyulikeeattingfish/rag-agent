# pgvector 使用要点

pgvector 是 PostgreSQL 的向量扩展，让向量和业务数据放在同一个库里，避免为了向量检索再引一个独立向量数据库。

## 操作符与 ops 类
- <=> 余弦距离（1 减余弦相似度），配 vector_cosine_ops
- <#> 负内积，配 vector_ip_ops
- <-> L2 距离，配 vector_l2_ops
排序时距离越小越相似，所以余弦距离要 ORDER BY ... ASC。
建索引语句形如 CREATE INDEX ON t USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64)。
**查询里的排序操作符必须和索引的 ops 类匹配，否则索引不会被使用。**

## 精确扫描与 ANN
不建索引时查询是精确的（暴力算距离），召回率 100%，但延迟随行数线性增长。
建了 ANN 索引后召回会有损失，需要拿精确扫描的结果作为 ground truth 来测 ANN 的 recall。

## 维度限制
vector 类型单列最多 2000 维，超过要用 halfvec（半精度，最多 4000 维）或做降维。HNSW 索引最高支持 2000 维。

## 与独立向量库的取舍
同一套 PostgreSQL 里做向量检索的好处是事务一致、少一个组件、可以直接 JOIN 业务表、能用元数据过滤；
代价是超大规模（亿级以上）时的性能和水平扩展不如专用向量数据库。中小规模场景 pgvector 完全够用。

