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


class DocumentChunk(BaseModel):
    chunk_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="The ID of the document chunk.",
    )
    content: str = Field(
        description="The chunk of text of a document",
    )
    original_span: tuple[int, int] = Field(
        description="The start and end position of the chunk in the original document",
    )
    hierarchy_level: int = Field(
        description="The level of hierarchy that this chunk belongs to",
    )
    parent_id: Optional[str] = Field(
        description="The ID of an optional upward related chunk",
    )
    embedding: Optional[list[float]] = Field(
        default=None,
        description="The embedding of the chunk",
    )


class ChunkedDocument(AbstractNamedDocument):
    document_metadata: Optional[dict] = Field(
        None,
        description="The optional document metadata dictionary",
    )
    chunks: list[DocumentChunk] = Field(
        description="The list of chunks of this document",
    )
