import base64

from tika import parser

from genie_flow_invoker.invoker.docproc.model import ChunkedDocument, RawDocumentFile, DocumentChunk

from genie_flow_invoker.invoker.docproc.parse import DocumentParseInvoker


def test_process_document(wilhelmus_path, tika_url):
    content = parser.from_file(
        wilhelmus_path,
        serverEndpoint=tika_url,
    )

    parsed_document = ChunkedDocument(
        filename=wilhelmus_path,
        chunks=[
            DocumentChunk(
                content=content["content"],
                hierarchy_level=0,
                parent_id=None,
                original_span=(0, len(content["content"])),
            )
        ],
        document_metadata=content["metadata"],
    )

    assert "Acrostycon" in parsed_document.chunks[0].content
    assert parsed_document.document_metadata["Content-Type"] == "application/pdf"


def test_parse_document(wilhelmus_path, tika_url):
    invoker = DocumentParseInvoker(tika_service_url=tika_url)

    with open(wilhelmus_path, "rb") as f:
        buffer = f.read()
        buffer_b64 = base64.b64encode(buffer).decode("ascii")

    input_document = RawDocumentFile(
        filename=wilhelmus_path,
        document_data=buffer_b64,
    )
    parsed_document_json = invoker.invoke(input_document.model_dump_json())

    assert "Acrostycon" in parsed_document_json
    assert "Content-Type" in parsed_document_json