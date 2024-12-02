from typing import Literal, Optional, Any, Iterable

import numpy as np
from numpy import dot, floating
from numpy.linalg import norm
from sortedcontainers import SortedList

from genie_flow_invoker.invoker.docproc.model import (
    DocumentChunk,
    DistanceMethodType,
    ChunkDistance,
)
from genie_flow_invoker.invoker.docproc.similarity_search.db import (
    VectorDB,
    ChunkVector,
)


_ONE = np.float32(1.0)


class SimilaritySearcher:

    def __init__(
        self,
        chunks: list[DocumentChunk],
        operation_level: Optional[int] = None,
        parent_strategy: Optional[Literal["include", "replace"]] = None,
    ):
        """
        A new `SimilaritySearcher` is initialized using a list of chunks and potentially
        a specified operation level and parent strategy.

        The operation level determines at which level in the chunk hierarchy are in scope
        for the search, defaulting to `None` which means: investigate all levels of the
        hierarchy.

        The parent strategy determines whether to include or replace the parent chunks
        of chunks that have been found. Including means their parents are added, replacing
        means that only the parents are returned.

        Note that any horizon filter is applied to the children first, before retrieving
        their parents. The same distance measure will be used to calculate the distance for
        the parents. When the parent strategy is `include`, both the parents and their
        children are returned, in order of distance to the search query.

        :param chunks: a list of chunks to search in
        :param operation_level: an optional operation level, defaults to `None`
        :param parent_strategy: an optional parent strategy, defaults to `None`
        """
        self._db = VectorDB(chunks)
        self._operation_level = operation_level
        self._parent_strategy = parent_strategy

    @staticmethod
    def method_cosine(v1: np.ndarray, v2: np.ndarray) -> float:
        return _ONE - dot(v1, v2) / (norm(v1) * norm(v2))

    @staticmethod
    def method_euclidian(v1: np.ndarray, v2: np.ndarray) -> floating[Any]:
        return norm(v1 - v2)

    @staticmethod
    def method_manhattan(v1: np.ndarray, v2: np.ndarray) -> float:
        return np.sum(np.absolute(v1 - v2))

    def _order_vectors(
        self,
        query_vector: np.ndarray,
        method: DistanceMethodType,
        chunk_vectors: Iterable[ChunkVector],
    ) -> SortedList[ChunkVector]:
        method_fn = self.__getattribute__(f"method_{method}")

        ordered_vectors: SortedList[ChunkVector] = SortedList(key=lambda x: x.distance)
        for chunk_vector in chunk_vectors:
            chunk_vector.distance = method_fn(chunk_vector.vector, query_vector)
            ordered_vectors.add(chunk_vector)

        return ordered_vectors

    @staticmethod
    def _find_horizon_cut_point(
        horizon: float, ordered_vectors: SortedList[ChunkVector]
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
        """
        From the objects database (list of chunks), return the similarities with a given
        query vector. Use the specified distance method (cosine, euclidean, manhattan) and
        apply filters for horizon and maximum number of results.

        Note that `horizon` will be applied to the children first, before retrieving
        their parents. Top is applied to the final results, potentially with parents
        included.

        The resulting list is ordered by distance to the search query.

        :param query_vector: The vector to search for
        :param method: the distance method to use
        :param horizon: an optional horizon for the distance, default `None`
        :param top: an optional maximum number of results to return, default `None`
        :return: an ordered (from low to high distance) list of `ChunkDistance` objects
        """
        if len(self._db) == 0 or top == 0:
            return []

        ordered_vectors = self._order_vectors(
            query_vector,
            method,
            self._db.get_vectors(operation_level=self._operation_level)
        )

        if horizon is not None:
            cut_point = self._find_horizon_cut_point(horizon, ordered_vectors)
            ordered_vectors = ordered_vectors[:cut_point]

        if self._parent_strategy is not None:
            parent_ids = {child.chunk.parent_id for child in ordered_vectors}
            parents = [self._db.get_vector(chunk_id) for chunk_id in parent_ids]
            ordered_parents = self._order_vectors(query_vector, method, parents)
            if self._parent_strategy == "include":
                # include the children with their parents
                ordered_parents.update(ordered_vectors)
            ordered_vectors = ordered_parents

        if top is not None:
            ordered_vectors = ordered_vectors[:top]

        return [
            ChunkDistance(
                chunk=ordered_vector.chunk,
                distance=float(ordered_vector.distance),
            )
            for ordered_vector in ordered_vectors
        ]
