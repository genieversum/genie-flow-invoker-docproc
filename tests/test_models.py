import base64

from tika import parser

from genie_flow_invoker.invoker.docproc.model import ParsedDocument, RawDocumentFile

from genie_flow_invoker.invoker.docproc.parse import DocumentParseInvoker


def test_process_document():
    document_path = "resources/Wilhelmus-van-Nassouwe.pdf"
    content = parser.from_file(document_path)

    parsed_document = ParsedDocument(
        filename=document_path,
        document_text=content["content"],
        document_metadata=content["metadata"],
    )

    assert "Acrostycon" in parsed_document.document_text
    assert parsed_document.document_metadata["Content-Type"] == "application/pdf"


def test_parse_document():
    document_path = "resources/Wilhelmus-van-Nassouwe.pdf"
    invoker = DocumentParseInvoker(
        tika_service_url="http://localhost:9998",
    )

    with open(document_path, "rb") as f:
        buffer = f.read()
        buffer_b64 = base64.b64encode(buffer).decode("ascii")

    input_document = RawDocumentFile(
        filename=document_path,
        document_data=buffer_b64,
    )
    parsed_document_json = invoker.invoke(input_document.model_dump_json())

    assert "Acrostycon" in parsed_document_json
    assert "Content-Type" in parsed_document_json