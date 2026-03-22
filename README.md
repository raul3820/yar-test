# Benchmark Comparison Results

## Models Compared

- **YARlabs/v5_Embedding_0.5B** (Lorentz distance, hyperbolic space, dim=64)
- **sentence-transformers/all-MiniLM-L6-v2** (cosine similarity, dim=384)

## Results

| Metric | YARlabs/v5_Embedding_0.5B | sentence-transformers/all-MiniLM-L6-v2 |
|--------|---------------------------|----------------------------------------|
| Recall@1 | 0.1077 | 0.3276 |
| Recall@5 | 0.3395 | 0.7918 |
| Recall@10 | 0.4619 | 0.9202 |
| MRR@10 | 0.2182 | 0.5201 |
| Corpus encoding | 4759.24s (79.3 min) | 122.57s (~2 min) |
| Query encoding | 52.78s | 2.68s |
| Total time | 4898.53s (81.6 min) | 128.12s (~2.1 min) |

## Dataset

- MS MARCO v1.1 test split
- Corpus: 77,897 unique passages
- Queries: 9,345