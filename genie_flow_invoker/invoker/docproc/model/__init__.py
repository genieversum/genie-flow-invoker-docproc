import base64
import io
import uuid
from typing import Optional

from pydantic import BaseModel, Field


class AbstractNamedDocument(BaseModel):
    filename: str = Field(
        description="The filename that has been given to this file",
    )


class RawDocumentFile(AbstractNamedDocument):
    document_data: str = Field(
        description="A base64 encoded version of the document data of this file",
    )

    @property
    def byte_io(self):
        buffer = base64.b64decode(self.document_data)
        return io.BytesIO(buffer)


class ParsedDocument(AbstractNamedDocument):
    document_text: str = Field(
        description="The text of the document content, as parsed by Tika",
    )
    document_metadata: dict = Field(
        description="The metadata of the document content as delivered by Tika",
    )


class DocumentChunk(AbstractNamedDocument):
    chunk_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="The ID of the document chunk.",
    )
    document_chunk: str = Field(
        description="The chunk of a document",
    )
    original_span: tuple[int, int] = Field(
        description="The start and end position of the chunk in the original document",
    )
    parent_id: Optional[str] = Field(
        default=None,
        description="The ID of an optional upward related chunk",
    )


class HierarchicalChunk(AbstractNamedDocument):
    chunk: DocumentChunk = Field(
        description="The chunk of a document",
    )
    children: list["HierarchicalChunk"] = Field(
        description="Optional child chunks of this parent"
    )


class ChunkedDocument(AbstractNamedDocument):
    chunks: list[DocumentChunk] = Field(
        description="The list of chunks of this document",
    )

    def in_hierarchy(self) -> list[HierarchicalChunk]:
        """
        Return this chunked document as a list of HierarchicalChunks.
        If a chunk has a parent, it will be recorded in the children list of
        the parent chunk.

        :return: a list of root chunks with their children
        """
        hierarchical_chunks = {
            document.chunk_id: HierarchicalChunk(
                filename=document.filename,
                chunk=document,
                children=list(),
            )
            for document in self.chunks
        }
        for id, c in hierarchical_chunks.items():
            if c.chunk.parent_id is None:
                continue
            hierarchical_chunks[c.chunk.parent_id].children.append(c)
        return [c for c in hierarchical_chunks.values() if c.chunk.parent_id is None]


class EmbeddedChunkedDocument(ChunkedDocument):
    embeddings: list[list[float]] = Field(
        default_factory=list,
        description="The embeddings of the chunks of this document",
    )
