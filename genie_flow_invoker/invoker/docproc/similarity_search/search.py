from typing import Literal, Optional

import numpy as np
from numpy import dot
from numpy.linalg import norm
from sortedcontainers import SortedList

from invoker.docproc.model import ChunkedDocument, DocumentChunk
from invoker.docproc.similarity_search import ChunkVector
from invoker.docproc.similarity_search.db import VectorDB

DistanceMethodType = Literal["cosine", "euclidean", "manhattan"]
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
        self._db = VectorDB(document)
        self._query_vector = np.array(query_vector, dtype=np.float32)
        self._operation_level = operation_level
        self._parent_strategy = parent_strategy

    def _order_vectors(
            self,
            method: DistanceMethodType,
    ) -> SortedList[ChunkVector]:
        method_fn = _DISTANCE_OPERATORS[method]

        ordered_vectors: SortedList[ChunkVector] = SortedList(key=lambda x: x.distance)
        for chunk_vector in self._db.get_vectors(operation_level=self._operation_level):
            chunk_vector.distance = method_fn(chunk_vector.distance, self._query_vector)
            ordered_vectors.append(chunk_vector)

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
            ordered_vectors.extend(
                self._db.get_vector(parent_id).chunk
                for parent_id in parent_ids
            )

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
