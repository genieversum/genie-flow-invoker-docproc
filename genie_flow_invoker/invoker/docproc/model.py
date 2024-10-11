import base64
import io

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
