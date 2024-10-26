from typing import Optional, NamedTuple, Literal

import numpy as np
from numpy import dot
from numpy.linalg import norm
from sortedcontainers import SortedList

from invoker.docproc.model import ChunkedDocument, DocumentChunk

ChunkVector = NamedTuple(
    "ChunkVector",
    [
        ("chunk", DocumentChunk),
        ("vector", np.ndarray),
        ("distance", Optional[float]),
    ]
)

DistanceMethodType = Literal["cosine", "euclidean", "manhattan"]

def _make_db(
        document: ChunkedDocument,
        operation_level: Optional[int] = None,
) -> list[ChunkVector]:
    return [
        ChunkVector(
            chunk=chunk,
            vector=np.array(chunk.embedding, dtype=np.float32),
        )
        for chunk in document.chunks
        if operation_level is None or chunk.hierarchy_level == operation_level
    ]


_DISTANCE_OPERATORS = dict(
    cosine=lambda a,b: np.float32(1.0) - dot(a, b) / (norm(a) * norm(b)),
    euclidean=lambda a,b: norm(a-b),
    manhattan=lambda a,b: np.sum(np.absolute(a-b))
)


class SimilaritySearcher:

    def __init__(
            self,
            document: ChunkedDocument,
            query_vector: np.ndarray,
            operation_level: Optional[int] = None,
            parent_strategy: Optional[Literal["include", "replace"]] = None,
    ):
        self._db = _make_db(document, operation_level)
        self._query_vector = np.array(query_vector, dtype=np.float32)
        self._parent_strategy = parent_strategy
        self._chunk_map = (
            {chunk.chunk_id: chunk for chunk in document.chunks}
            if parent_strategy is not None
            else None
        )

    def _order_vectors(self, method: DistanceMethodType) -> SortedList[ChunkVector]:
        method_fn = _DISTANCE_OPERATORS[method]

        ordered_vectors: SortedList[ChunkVector] = SortedList(key=lambda x: x.distance)
        for chunk_vector in self._db:
            ordered_vectors.append(
                ChunkVector(
                    chunk=chunk_vector.chunk,
                    vector=chunk_vector.vector,
                    distance=method_fn(self._query_vector, chunk_vector.vector),
                )
            )

        return ordered_vectors

    def calculate_similarities(
            self,
            horizon: Optional[float],
            top: Optional[int],
            method: DistanceMethodType,
    ) -> list[DocumentChunk]:
        ordered_vectors = self._order_vectors(method)

        if self._parent_strategy is not None:
            parent_ids = {child.chunk.parent_id for child in ordered_vectors}

            if self._parent_strategy == "replace":
                ordered_vectors.clear()
            ordered_vectors.extend(self._chunk_map[parent_id] for parent_id in parent_ids)

        if top is not None:
            ordered_vectors = ordered_vectors[:top]

        if horizon is not None:
            cut_point = len(ordered_vectors)
            for i, chunk_vector in enumerate(ordered_vectors):
                if chunk_vector.distance >= horizon:
                    cut_point = i
                    break

            ordered_vectors = ordered_vectors[:cut_point]

        return list(ordered_vectors)
