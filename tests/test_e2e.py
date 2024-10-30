import base64

from invoker.docproc.chunk import LexicalDensitySplitInvoker
from invoker.docproc.clean import DocumentCleanInvoker
from invoker.docproc.embed import EmbedInvoker
from invoker.docproc.model import RawDocumentFile, ChunkedDocument
from invoker.docproc.parse import DocumentParseInvoker


def test_parse_to_embedding(wilhelmus_path, tika_url, t2v_url):
    with open(wilhelmus_path, "rb") as f:
        buffer = f.read()
        buffer_b64 = base64.b64encode(buffer).decode("ascii")

    input_document = RawDocumentFile(
        filename=wilhelmus_path,
        document_data=buffer_b64,
    )
    invoker = DocumentParseInvoker(tika_service_url=tika_url)
    parsed_document_json = invoker.invoke(input_document.model_dump_json())
    parsed_document = ChunkedDocument.model_validate_json(parsed_document_json)

    invoker = DocumentCleanInvoker(
        clean_multiple_newlines=True,
        clean_multiple_spaces=False,
        clean_tabs=True,
        clean_numbers=True,
        special_term_replacements={},
        tokenize_detokenize=True,
    )
    cleaned_document_json = invoker.invoke(parsed_document.model_dump_json())
    cleaned_document = ChunkedDocument.model_validate_json(cleaned_document_json)

    invoker =  LexicalDensitySplitInvoker(
        min_words=16,
        max_words=32,
        overlap=4,
        target_density=0.75,
        strategy="shortest",
        operation_level=0,
    )
    chunked_document_json = invoker.invoke(cleaned_document.model_dump_json())
    chunked_document = ChunkedDocument.model_validate_json(chunked_document_json)

    print(chunked_document.model_dump_json(indent=2))

    invoker = EmbedInvoker(
        text2vec_url=t2v_url,
        pooling_strategy="mean",
    )
    embedded_document_json = invoker.invoke(chunked_document.model_dump_json())
    embedded_document = ChunkedDocument.model_validate_json(embedded_document_json)

    print(embedded_document.model_dump_json(indent=2))