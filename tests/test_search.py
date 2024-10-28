from collections.abc import Callable

import numpy as np
from numpy import dot
from numpy.linalg import norm
import pytest

from invoker.docproc.model import DocumentChunk, DistanceMethodType, SimilaritySearch, \
    SimilarityResults
from invoker.docproc.similarity_search import SimilaritySearchInvoker
from invoker.docproc.similarity_search.search import SimilaritySearcher


@pytest.fixture
def hamlet_chunks_with_vectors(hamlet_content_cleaned):
    texts = [
        "to be or not to be",
        "More matter with less art",
        "O, what a noble mind is here o'erthrown!",
        "How now, a rat? Dead for a ducat, dead.",
        "the rest is silence",
        "I am dead, Horatio.",
    ]
    return [
        DocumentChunk(
            content=text,
            parent_id=None,
            original_span=(0, len(text)),
            hierarchy_level=0,
            embedding=[0.5 if j <= i else 0.0 for j in range(len(texts))],
        )
        for i, text in enumerate(texts)
    ]


def test_search_cosine(hamlet_chunks_with_vectors):
    query_vector = np.array([1] * len(hamlet_chunks_with_vectors))

    similarity_searcher = SimilaritySearcher(
        chunks=hamlet_chunks_with_vectors,
    )

    similarities = similarity_searcher.calculate_similarities(
        query_vector=query_vector,
        method="cosine",
        horizon=None,
        top=None,
    )

    assert len(similarities) == len(hamlet_chunks_with_vectors)
    previous_distance = None
    for similarity in similarities:
        if previous_distance is not None:
            assert sum(similarity.chunk.embedding) < previous_distance
        previous_distance = sum(similarity.chunk.embedding)


def test_similarity_top(hamlet_chunks_with_vectors):
    query_vector = np.array([1] * len(hamlet_chunks_with_vectors))

    similarity_searcher = SimilaritySearcher(
        chunks=hamlet_chunks_with_vectors,
    )

    similarities = similarity_searcher.calculate_similarities(
        query_vector=query_vector,
        method="cosine",
        top=2,
        horizon=None,
    )

    assert len(similarities) == 2


def test_similarity_horizon(hamlet_chunks_with_vectors):
    query_vector = np.array([1] * len(hamlet_chunks_with_vectors))

    similarity_searcher = SimilaritySearcher(
        chunks=hamlet_chunks_with_vectors,
    )

    similarities = similarity_searcher.calculate_similarities(
        query_vector=query_vector,
        method="cosine",
        top=None,
        horizon=0.2,
    )

    assert len(similarities) == 3


def test_similarity_top_then_horizon(hamlet_chunks_with_vectors):
    query_vector = np.array([1] * len(hamlet_chunks_with_vectors))

    similarity_searcher = SimilaritySearcher(
        chunks=hamlet_chunks_with_vectors,
    )

    similarities = similarity_searcher.calculate_similarities(
        query_vector=query_vector,
        method="cosine",
        top=2,
        horizon=0.8,
    )

    assert len(similarities) == 2


def test_similarity_horizon_then_top(hamlet_chunks_with_vectors):
    query_vector = np.array([1] * len(hamlet_chunks_with_vectors))

    similarity_searcher = SimilaritySearcher(
        chunks=hamlet_chunks_with_vectors,
    )

    similarities = similarity_searcher.calculate_similarities(
        query_vector=query_vector,
        method="cosine",
        top=2,
        horizon=0.8,
    )

    assert len(similarities) == 2


def test_similarity_no_docs():
    similarity_searcher = SimilaritySearcher(
        chunks=[],
    )
    similarities = similarity_searcher.calculate_similarities(
        query_vector=np.array([1, 1, 1]),
        method="cosine",
        top=None,
        horizon=None,
    )

    assert similarities == []


def test_similarity_distance_methods(hamlet_chunks_with_vectors):
    query_vector = np.array([1] * len(hamlet_chunks_with_vectors))

    similarity_searcher = SimilaritySearcher(
        chunks=hamlet_chunks_with_vectors,
    )

    one = np.float32(1.0)
    method_fn: dict[DistanceMethodType, Callable[[np.ndarray, np.ndarray], float]] = {
        "euclidian": lambda v1, v2: norm(v1 - v2),
        "cosine": lambda v1, v2: one - dot(v1, v2) / (norm(v1) * norm(v2)),
        "manhattan": lambda v1, v2: np.sum(np.abs(v1 - v2))
    }

    for method, dist_fn in method_fn.items():
        expected_order = [
            (
                chunk.chunk_id,
                dist_fn(query_vector, np.array(chunk.embedding))
            )
            for chunk in hamlet_chunks_with_vectors
        ]
        expected_order.sort(key=lambda x: x[1])
        similarities = similarity_searcher.calculate_similarities(
            query_vector=query_vector,
            method=method,
            top=None,
            horizon=None,
        )
        for i, (expected_chunk_id, _) in enumerate(expected_order):
            assert (
                expected_chunk_id == similarities[i].chunk.chunk_id
            ), f"order fails for method {method}, index {i}"


def test_similarity_invoking(hamlet_chunks_with_vectors):
    query_vector = [1] * len(hamlet_chunks_with_vectors)

    search_query = SimilaritySearch(
        filename="Hamlet.txt",
        chunks=hamlet_chunks_with_vectors,
        query_embedding=query_vector,
        method="cosine",
    )
    search_query_json = search_query.model_dump_json(indent=2)

    invoker = SimilaritySearchInvoker()
    search_result_json = invoker.invoke(search_query_json)
    search_result = SimilarityResults.model_validate_json(search_result_json)

    assert len(search_result.chunk_distances) == len(hamlet_chunks_with_vectors)


def test_reads_config():
    config = dict(
        operation_level=2,
        horizon=0.6,
        top=15,
        parent_strategy="include",
        method="cosine",
    )
    invoker = SimilaritySearchInvoker.from_config(config)

    assert invoker.operation_level == 2
    assert invoker.horizon == 0.6
    assert invoker.top == 15
    assert invoker.parent_strategy == "include"
    assert invoker.method == "cosine"
