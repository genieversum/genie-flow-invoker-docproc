from typing import Literal, Optional, Any, Iterable

import numpy as np
from loguru import logger
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

    def _retrieve_parents(
            self,
            ordered_children: SortedList[ChunkVector],
    ) -> list[ChunkVector]:
        """
        Retrieve the parent chunks of the given vectors. If no parent exists, validate if
        they are ultimate parent or not. If not, raise an error. If they are ultimate parents,
        they are added to the list of parents.

        :param ordered_children: a list of vectors to retrieve parents for
        :return: a list of chunk vectors of the parents of all children
        """
        parent_ids: set[Optional[str]] = {
            child.chunk.parent_id
            for child in ordered_children
        }

        if None in parent_ids:
            parent_ids.remove(None)

            orphans: list[ChunkVector] = [
                child
                for child in ordered_children
                if child.chunk.parent_id is None
            ]
            if any(orphan.chunk.hierarchy_level != 0 for orphan in orphans):
                logger.debug(
                    "following chunks are not on top of the hierarchy but still "
                    "have no parents: {orphans}",
                    orphans=[
                        orphan.chunk
                        for orphan in orphans
                        if orphan.chunk.hierarchy_level != 0
                    ],
                )
                logger.error(
                    "structural issue with chunk hierarchy; some chunks do not have a "
                    "parent but are not at the highest hierarchical level"
                )
                raise ValueError("structural issue with chunk hierarchy")

            logger.warning(
                "{nr_ultimate_parents} ultimate parent(s) will not be subjected "
                "to parent strategy {parent_strategy}",
                nr_ultimate_parents=len(orphans),
                parent_strategy=self._parent_strategy,
            )
            parent_ids.update({orphan.chunk.chunk_id for orphan in orphans})

        try:
            return [self._db.get_vector(chunk_id) for chunk_id in parent_ids]
        except KeyError as e:
            logger.error(
                "structural issue with chunk hierarchy; "
                "child refers to non-existent parent chunk"
            )
            raise e

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
        logger.debug(
            "Calculating similarities with settings method {method}, "
            "horizon {horizon}, top {top}",
            method=method,
            horizon=horizon,
            top=top,
        )
        if len(self._db) == 0 or top == 0:
            logger.warning(
                "Searching will have no results; "
                "database has {db_size} chunks; top argument is set to {top}",
                db_size=len(self._db),
                top=top,
            )
            return []

        ordered_vectors = self._order_vectors(
            query_vector,
            method,
            self._db.get_vectors(operation_level=self._operation_level)
        )
        logger.debug("found {nr_chunks} from the database", nr_chunks=len(ordered_vectors))

        if horizon is not None:
            cut_point = self._find_horizon_cut_point(horizon, ordered_vectors)
            ordered_vectors = ordered_vectors[:cut_point]
            logger.debug("cut because of horizon, at point {cut_point}", cut_point=cut_point)

        if self._parent_strategy is not None:
            parents = self._retrieve_parents(ordered_vectors)
            ordered_parents = self._order_vectors(query_vector, method, parents)
            logger.debug("found {nr_parents} from the children", nr_parents=len(ordered_parents))

            match self._parent_strategy:
                case "include":
                    logger.debug(
                        "adding {nr_parents} parents to {nr_children} children",
                        nr_children=len(ordered_vectors),
                        nr_parents=len(ordered_parents),
                    )
                    child_ids = {chunk_vector.chunk.chunk_id for chunk_vector in ordered_vectors}
                    for parent in parents:
                        if parent.chunk.chunk_id not in child_ids:
                            ordered_vectors.add(parent)
                case "replace":
                    logger.debug(
                        "replacing {nr_children} children with {nr_parents} parents",
                        nr_children=len(ordered_vectors),
                        nr_parents=len(ordered_parents),
                    )
                    ordered_vectors = ordered_parents
                case _:
                    logger.error(
                        "parent strategy {parent_strategy} not recognized",
                        parent_strategy=self._parent_strategy,
                    )
                    raise ValueError("invalid parent strategy")

        if top is not None:
            ordered_vectors = ordered_vectors[:top]
            logger.debug("limiting to {top} results", top=top)

        result = [
            ChunkDistance(
                chunk=ordered_vector.chunk,
                distance=float(ordered_vector.distance),
            )
            for ordered_vector in ordered_vectors
        ]
        logger.info(
            "found {nr_chunks} using similarity search method",
            nr_chunks=len(result),
            method=method,
        )
        return result