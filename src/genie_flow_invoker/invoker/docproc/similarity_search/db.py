from dataclasses import dataclass
from typing import Optional

import numpy as np

from genie_flow_invoker.invoker.docproc.model import ChunkedDocument, DocumentChunk


@dataclass
class ChunkVector:
    chunk: DocumentChunk
    vector: np.ndarray
    distance: Optional[float]


class VectorDB:
    def __init__(self, chunks: list[DocumentChunk]):
        self._chunk_vectors = [
            ChunkVector(
                chunk=chunk,
                vector=np.array(chunk.embedding, dtype=np.float32),
                distance=None,
            )
            for chunk in chunks
        ]
        self._chunk_id_index: dict[str, ChunkVector] = dict()
        self._level_index: dict[int, list[ChunkVector]] = dict()

        for chunk_vector in self._chunk_vectors:
            self._chunk_id_index[chunk_vector.chunk.chunk_id] = chunk_vector
            if chunk_vector.chunk.hierarchy_level not in self._level_index:
                self._level_index[chunk_vector.chunk.hierarchy_level] = []
            self._level_index[chunk_vector.chunk.hierarchy_level].append(chunk_vector)

    def __len__(self):
        return len(self._chunk_vectors)

    def get_vector(self, chunk_id: str) -> ChunkVector:
        return self._chunk_id_index[chunk_id]

    def get_vectors(self, operation_level: Optional[int] = None) -> list[ChunkVector]:
        if operation_level is None:
            return self._chunk_vectors
        return self._level_index[operation_level]
