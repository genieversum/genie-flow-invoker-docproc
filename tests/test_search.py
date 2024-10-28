import numpy as np
import pytest

from invoker.docproc.chunk import LexicalDensitySplitter
from invoker.docproc.model import DocumentChunk, ChunkedDocument
from invoker.docproc.similarity_search.search import SimilaritySearcher


@pytest.fixture
def hamlet_chunks_with_vectors(hamlet_content_cleaned):
    texts = [
        "to be or not to be",
        "More matter with less art",
        "O, what a noble mind is here o'erthrown!",
        "How now, a rat? Dead for a ducat, dead.",
        "the rest is silence",
        "I am dead, Horatio."
    ]
    return ChunkedDocument(
        filename="Hamlet.txt",
        chunks=[
            DocumentChunk(
                content=text,
                parent_id=None,
                original_span=(0, len(text)),
                hierarchy_level=0,
                embedding=[
                    0.5 if j <= i else 0.0
                    for j in range(len(texts))
                ]
            )
            for i, text in enumerate(texts)
        ],
    )


def test_search_cosine(hamlet_chunks_with_vectors):
    query_vector = np.array([1] * len(hamlet_chunks_with_vectors.chunks))

    similarity_searcher = SimilaritySearcher(
        document=hamlet_chunks_with_vectors,
        query_vector=query_vector,
    )

    similarities = similarity_searcher.calculate_similarities(
        method="cosine",
        horizon=None,
        top=None,
    )

    assert len(similarities) == len(hamlet_chunks_with_vectors.chunks)
    previous_distance = None
    for similarity in similarities:
        if previous_distance is not None:
            assert sum(similarity.embedding) < previous_distance
        previous_distance = sum(similarity.embedding)


def test_similarity_top(hamlet_chunks_with_vectors):
    query_vector = np.array([1] * len(hamlet_chunks_with_vectors.chunks))

    similarity_searcher = SimilaritySearcher(
        document=hamlet_chunks_with_vectors,
        query_vector=query_vector,
    )

    similarities = similarity_searcher.calculate_similarities(
        method="cosine",
        top=2,
        horizon=None,
    )

    assert len(similarities) == 2


def test_similarity_horizon(hamlet_chunks_with_vectors):
    query_vector = np.array([1] * len(hamlet_chunks_with_vectors.chunks))

    similarity_searcher = SimilaritySearcher(
        document=hamlet_chunks_with_vectors,
        query_vector=query_vector,
    )

    similarities = similarity_searcher.calculate_similarities(
        method="cosine",
        top=None,
        horizon=0.2,
    )

    assert len(similarities) == 3


def test_similarity_top_horizon(hamlet_chunks_with_vectors):
    query_vector = np.array([1] * len(hamlet_chunks_with_vectors.chunks))

    similarity_searcher = SimilaritySearcher(
        document=hamlet_chunks_with_vectors,
        query_vector=query_vector,
    )

    similarities = similarity_searcher.calculate_similarities(
        method="cosine",
        top=2,
        horizon=0.2,
    )

    assert len(similarities) == 2
