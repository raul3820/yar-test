#!/usr/bin/env python3
"""
Benchmark script for YARlabs/v5_Embedding_0.5B model on LegalQAEval dataset.
Uses Lorentz distance for similarity computation in hyperbolic space.
"""

import random
import time
import logging
from typing import List, Tuple, Any

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def lorentz_dist(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Computes the exact Hyperbolic distance between two batches of Lorentz vectors.
    """
    u_0, u_x = u[..., 0:1], u[..., 1:]
    v_0, v_x = v[..., 0:1], v[..., 1:]

    inner_product = -u_0 * v_0 + (u_x * v_x).sum(dim=-1, keepdim=True)
    inner_product = torch.min(inner_product, torch.tensor(-1.0, device=u.device))
    return torch.acosh(-inner_product).squeeze(-1)


def prepare_data(
    dataset: Any, corpus_size: int = 0
) -> Tuple[List[str], List[str], List[List[str]]]:
    """Prepare corpus, queries, and ground truths."""
    random.seed(42)

    all_texts = [item["text"] for item in dataset]
    unique_texts = sorted(list(set(all_texts)))

    if corpus_size > 0 and len(unique_texts) > corpus_size:
        unique_texts = sorted(random.sample(unique_texts, corpus_size))

    queries = {}
    for item in dataset:
        if item["answers"]:
            q_text = item["question"]
            if item["text"] in unique_texts:
                if q_text not in queries:
                    queries[q_text] = []
                queries[q_text].append(item["text"])

    queries = {q: texts for q, texts in queries.items() if texts}

    if not queries:
        logger.error("No valid queries after filtering")
        return [], [], []

    query_list = list(queries.keys())
    ground_truths = list(queries.values())

    logger.info(f"Prepared {len(unique_texts)} unique texts, {len(queries)} queries")

    return unique_texts, query_list, ground_truths


def batch_encode(
    texts: List[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    target_dim: int = 64,
    batch_size: int = 32,
    device: str = "cuda",
) -> torch.Tensor:
    """Encode texts in batches using the YarEmbedding model."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            embeddings = model(**inputs, target_dim=target_dim)

        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def compute_similarity_lorentz(
    queries: torch.Tensor, corpus: torch.Tensor
) -> np.ndarray:
    """
    Compute similarity scores using Lorentz distance.
    For retrieval, we use negative distance (closer = higher similarity).
    """
    device = queries.device

    # Process in batches to avoid OOM
    batch_size = 32
    scores = []

    for i in range(0, queries.shape[0], batch_size):
        batch_queries = queries[i : i + batch_size].to(device)
        batch_scores = []

        for j in range(0, corpus.shape[0], batch_size):
            batch_corpus = corpus[j : j + batch_size].to(device)
            distances = lorentz_dist(
                batch_queries[:, None, :], batch_corpus[None, :, :]
            )
            # Convert distance to similarity (negative distance)
            batch_scores.append(-distances.float().cpu().numpy())

        scores.append(np.concatenate(batch_scores, axis=1))

    return np.concatenate(scores, axis=0)


def compute_metrics(
    scores: np.ndarray,
    unique_texts: List[str],
    query_list: List[str],
    ground_truths: List[List[str]],
    top_k_list: List[int] = [1, 5, 10],
) -> dict:
    """Compute MRR and Recall metrics."""
    results = {k: 0 for k in top_k_list}
    mrr = 0

    for i in range(len(query_list)):
        query_scores = scores[i]
        retrieved_indices = np.argsort(query_scores)[::-1]  # Descending order

        relevant_texts = set(ground_truths[i])

        # Recall@k
        for k in top_k_list:
            if any(
                unique_texts[idx] in relevant_texts for idx in retrieved_indices[:k]
            ):
                results[k] += 1

        # MRR
        for rank, doc_id in enumerate(retrieved_indices):
            if unique_texts[doc_id] in relevant_texts:
                mrr += 1.0 / (rank + 1)
                break

    n_queries = len(query_list)
    metrics = {f"Recall@{k}": results[k] / n_queries for k in top_k_list}
    metrics["MRR@10"] = mrr / n_queries

    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark YarEmbedding model")
    parser.add_argument(
        "--target_dim", type=int, default=64, help="Lorentz embedding dimension"
    )
    parser.add_argument(
        "--corpus_size", type=int, default=0, help="Corpus size (0 = full dataset)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for encoding"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    logger.info(f"Target dimension: {args.target_dim}")
    logger.info(f"Corpus size: {args.corpus_size if args.corpus_size else 'full'}")
    logger.info(f"Device: {args.device}")

    # Load model
    model_id = "YARlabs/v5_Embedding_0.5B"
    logger.info(f"Loading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval()
    model = model.to(args.device)

    logger.info("Model loaded successfully!")

    # Load dataset
    logger.info("Loading LegalQAEval dataset...")
    try:
        val_dataset = load_dataset("isaacus/LegalQAEval")
        dataset = concatenate_datasets([val_dataset["val"], val_dataset["test"]])
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Prepare data
    unique_texts, query_list, ground_truths = prepare_data(
        dataset, corpus_size=args.corpus_size
    )

    if not unique_texts or not query_list:
        logger.error("No data to evaluate")
        return

    # Encode corpus
    logger.info("Encoding corpus...")
    t0 = time.time()
    corpus_embeddings = batch_encode(
        unique_texts,
        model,
        tokenizer,
        target_dim=args.target_dim,
        batch_size=args.batch_size,
        device=args.device,
    )
    corpus_encode_time = time.time() - t0
    logger.info(
        f"Corpus encoded in {corpus_encode_time:.2f}s, shape: {corpus_embeddings.shape}"
    )

    # Encode queries
    logger.info("Encoding queries...")
    t1 = time.time()
    query_embeddings = batch_encode(
        query_list,
        model,
        tokenizer,
        target_dim=args.target_dim,
        batch_size=args.batch_size,
        device=args.device,
    )
    query_encode_time = time.time() - t1
    logger.info(
        f"Queries encoded in {query_encode_time:.2f}s, shape: {query_embeddings.shape}"
    )

    # Compute similarity using Lorentz distance
    logger.info("Computing similarities using Lorentz distance...")
    t2 = time.time()
    scores = compute_similarity_lorentz(query_embeddings, corpus_embeddings)
    similarity_time = time.time() - t2
    logger.info(f"Similarity computation took {similarity_time:.2f}s")

    # Compute metrics
    metrics = compute_metrics(scores, unique_texts, query_list, ground_truths)

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Target dimension: {args.target_dim}")
    print(f"Corpus size: {len(unique_texts)}")
    print(f"Number of queries: {len(query_list)}")
    print("-" * 60)
    print("RETRIEVAL METRICS:")
    for k in [1, 5, 10]:
        print(f"  Recall@{k}: {metrics[f'Recall@{k}']:.4f}")
    print(f"  MRR@10: {metrics['MRR@10']:.4f}")
    print("-" * 60)
    print("TIMING:")
    print(f"  Corpus encoding: {corpus_encode_time:.2f}s")
    print(f"  Query encoding: {query_encode_time:.2f}s")
    print(f"  Similarity computation: {similarity_time:.2f}s")
    print(
        f"  Total time: {corpus_encode_time + query_encode_time + similarity_time:.2f}s"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
