#!/usr/bin/env python3
"""
Benchmark script for sentence-transformers/all-MiniLM-L6-v2 model on MS MARCO dataset.
Uses cosine similarity for retrieval.
"""

import random
import time
import logging
from typing import List, Tuple, Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_data_ms_marco(
    dataset: Any, corpus_size: int = 0
) -> Tuple[List[str], List[str], List[List[str]]]:
    """Prepare corpus, queries, and ground truths from MS MARCO dataset."""
    random.seed(42)

    # Extract all passages from all samples
    all_texts = []
    for item in dataset:
        passages = item.get("passages", {})
        passage_texts = passages.get("passage_text", [])
        all_texts.extend(passage_texts)

    unique_texts = sorted(list(set(all_texts)))

    if corpus_size > 0 and len(unique_texts) > corpus_size:
        unique_texts = sorted(random.sample(unique_texts, corpus_size))

    # Build queries and ground truths
    # In MS MARCO, passages with is_selected=1 are relevant
    queries = {}
    for item in dataset:
        q_text = item["query"]
        passages = item.get("passages", {})
        passage_texts = passages.get("passage_text", [])
        is_selected = passages.get("is_selected", [])

        # Get relevant passages for this query
        relevant_texts = []
        for idx, (text, selected) in enumerate(zip(passage_texts, is_selected)):
            if selected == 1 and text in unique_texts:
                relevant_texts.append(text)

        if relevant_texts and q_text:
            if q_text not in queries:
                queries[q_text] = []
            queries[q_text].extend(relevant_texts)

    # Deduplicate ground truths per query
    queries = {q: list(set(texts)) for q, texts in queries.items()}

    if not queries:
        logger.error("No valid queries after filtering")
        return [], [], []

    query_list = list(queries.keys())
    ground_truths = list(queries.values())

    logger.info(f"Prepared {len(unique_texts)} unique texts, {len(queries)} queries")

    return unique_texts, query_list, ground_truths


def batch_encode(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = 128,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode texts in batches using the sentence-transformer model."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # For cosine similarity
    )
    return embeddings


def compute_similarity_cosine(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity scores.
    With normalized embeddings, this is just dot product.
    """
    # Efficient computation using matrix multiplication
    scores = np.matmul(queries, corpus.T)
    return scores


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

    parser = argparse.ArgumentParser(description="Benchmark SentenceTransformer model")
    parser.add_argument(
        "--corpus_size", type=int, default=0, help="Corpus size (0 = full dataset)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for encoding"
    )
    args = parser.parse_args()

    logger.info(f"Corpus size: {args.corpus_size if args.corpus_size else 'full'}")
    logger.info(f"Batch size: {args.batch_size}")

    # Load model
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    logger.info(f"Loading model: {model_id}")

    model = SentenceTransformer(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    model.to(device)

    logger.info("Model loaded successfully!")

    # Load dataset
    logger.info("Loading microsoft/ms_marco 'v1.1' dataset (test split)...")
    try:
        dataset = load_dataset("microsoft/ms_marco", "v1.1", split="test")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Prepare data using MS MARCO format
    unique_texts, query_list, ground_truths = prepare_data_ms_marco(
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
        batch_size=args.batch_size,
        show_progress=True,
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
        batch_size=args.batch_size,
        show_progress=True,
    )
    query_encode_time = time.time() - t1
    logger.info(
        f"Queries encoded in {query_encode_time:.2f}s, shape: {query_embeddings.shape}"
    )

    # Compute similarity using cosine similarity
    logger.info("Computing similarities using cosine similarity...")
    t2 = time.time()
    scores = compute_similarity_cosine(query_embeddings, corpus_embeddings)
    similarity_time = time.time() - t2
    logger.info(f"Similarity computation took {similarity_time:.2f}s")

    # Compute metrics
    metrics = compute_metrics(scores, unique_texts, query_list, ground_truths)

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Embedding dimension: 384")
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
