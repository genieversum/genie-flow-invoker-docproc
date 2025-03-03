from pydantic import Field

from genie_flow_invoker.doc_proc import (
    AbstractNamedDocument,
    DocumentChunk,
    ChunkDistance,
    SimilaritySearchRequest,
)


class SimilaritySearch(SimilaritySearchRequest):
    chunks: list[DocumentChunk] = Field(
        description="The list of chunks of this document",
    )


class SimilarityResults(AbstractNamedDocument):
    chunk_distances: list[ChunkDistance] = Field(
        description="The list of chunks and their distances towards search query",
    )
