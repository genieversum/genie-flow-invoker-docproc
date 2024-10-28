from typing import Literal, Optional, Any

import numpy as np
from numpy import dot, floating
from numpy.linalg import norm
from sortedcontainers import SortedList

from invoker.docproc.model import DocumentChunk, DistanceMethodType, ChunkDistance
from invoker.docproc.similarity_search.db import VectorDB, ChunkVector


_ONE = np.float32(1.0)


class SimilaritySearcher:

    def __init__(
            self,
            chunks: list[DocumentChunk],
            operation_level: Optional[int] = None,
            parent_strategy: Optional[Literal["include", "replace"]] = None,
    ):
        self._db = VectorDB(chunks)
        self._operation_level = operation_level
        self._parent_strategy = parent_strategy

    @staticmethod
    def method_cosine(v1: np.ndarray, v2: np.ndarray) -> float:
        return _ONE - dot(v1, v2) / (norm(v1) * norm(v2))

    @staticmethod
    def method_euclidian(v1: np.ndarray, v2: np.ndarray) -> floating[Any]:
        return norm(v1-v2)

    @staticmethod
    def method_manhattan(v1: np.ndarray, v2: np.ndarray) -> float:
        return np.sum(np.absolute(v1-v2))

    def _order_vectors(
            self,
            query_vector: np.ndarray,
            method: DistanceMethodType,
    ) -> SortedList[ChunkVector]:
        method_fn = self.__getattribute__(f"method_{method}")

        ordered_vectors: SortedList[ChunkVector] = SortedList(key=lambda x: x.distance)
        for chunk_vector in self._db.get_vectors(operation_level=self._operation_level):
            chunk_vector.distance = method_fn(chunk_vector.vector, query_vector)
            ordered_vectors.add(chunk_vector)

        return ordered_vectors

    def _introduce_parents(self, ordered_vectors: SortedList[ChunkVector]) -> None:
        parent_ids = {child.chunk.parent_id for child in ordered_vectors}

        if self._parent_strategy == "replace":
            ordered_vectors.clear()
        ordered_vectors.extend(
            self._db.get_vector(parent_id).chunk
            for parent_id in parent_ids
        )

    @staticmethod
    def _find_horizon_cut_point(
            horizon: float,
            ordered_vectors: SortedList[ChunkVector]
    ) -> int:
        if ordered_vectors[-1].distance < horizon:
            return len(ordered_vectors)

        for i, chunk_vector in enumerate(ordered_vectors):
            if chunk_vector.distance >= horizon:
                return i

    def calculate_similarities(
            self,
            query_vector: np.ndarray,
            method: DistanceMethodType,
            horizon: Optional[float] = None,
            top: Optional[int] = None,
    ) -> list[ChunkDistance]:
        if len(self._db) == 0 or top == 0:
            return []

        ordered_vectors = self._order_vectors(query_vector, method)

        if self._parent_strategy is not None:
            self._introduce_parents(ordered_vectors)

        if top is not None:
            ordered_vectors = ordered_vectors[:top]

        if horizon is not None:
            cut_point = self._find_horizon_cut_point(horizon, ordered_vectors)
            ordered_vectors = ordered_vectors[:cut_point]

        return [
            ChunkDistance(
                chunk=ordered_vector.chunk,
                distance=float(ordered_vector.distance),
            )
            for ordered_vector in ordered_vectors
        ]
