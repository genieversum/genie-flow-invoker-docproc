import json

from genie_flow_invoker.invoker.docproc.chunk import LexicalDensitySplitInvoker
from genie_flow_invoker.invoker.docproc.clean import DocumentCleanInvoker
from genie_flow_invoker.invoker.docproc.embed import EmbedInvoker
from genie_flow_invoker.invoker.docproc.model import (
    SimilaritySearch,
    SimilarityResults,
)
from genie_flow_invoker.doc_proc import DocumentChunk, ChunkedDocument
from genie_flow_invoker.invoker.docproc.similarity_search import SimilaritySearchInvoker


def test_parse_to_search(pa_text, t2v_url):
    parsed_document = ChunkedDocument(
        filename="about-pa.txt",
        chunks=[
            DocumentChunk(
                content=pa_text,
                original_span=(0, len(pa_text)),
                parent_id=None,
                hierarchy_level=0,
            )
        ]
    )
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

    assert cleaned_document is not None
    assert len(cleaned_document.chunks) == 1

    invoker =  LexicalDensitySplitInvoker(
        min_words=8,
        max_words=16,
        overlap=4,
        target_density=0.6,
        strategy="shortest",
        operation_level=0,
    )
    chunked_document_json = invoker.invoke(cleaned_document.model_dump_json())
    chunked_document = ChunkedDocument.model_validate_json(chunked_document_json)

    assert chunked_document is not None
    assert len(chunked_document.chunks) > 1
    hierarchies = {chunk.hierarchy_level for chunk in chunked_document.chunks}
    assert hierarchies == {0, 1}

    invoker = EmbedInvoker(
        text2vec_url=t2v_url,
        pooling_strategy="masked_mean",
    )
    embedded_document_json = invoker.invoke(chunked_document.model_dump_json())
    embedded_document = ChunkedDocument.model_validate_json(embedded_document_json)

    for chunk in embedded_document.chunks:
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 384

    query_text = "what makes people happy"
    query_embedding_json = invoker.invoke(query_text)
    query_embedding = json.loads(query_embedding_json)

    invoker = SimilaritySearchInvoker()

    search_query = SimilaritySearch(
        filename=embedded_document.filename,
        chunks=embedded_document.chunks,
        query_embedding=query_embedding,
        operation_level=1,
        horizon=0.75,
        top=5,
    )
    search_query_json = search_query.model_dump_json()
    search_results_json = invoker.invoke(search_query_json)
    search_results = SimilarityResults.model_validate_json(search_results_json)

    for chunk_distance in search_results.chunk_distances:
        print(f"{chunk_distance.distance} -- {chunk_distance.chunk.content}")

    assert search_results is not None
    assert 0 < len(search_results.chunk_distances) <= 5
    assert (search_results.chunk_distances[0].distance <
            search_results.chunk_distances[-1].distance)
