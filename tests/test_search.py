from invoker.docproc.chunk import LexicalDensitySplitter
from invoker.docproc.model import DocumentChunk, ChunkedDocument
from invoker.docproc.similarity_search.search import SimilaritySearcher


def test_search(hamlet_content_cleaned):
    document_chunk = DocumentChunk(
        content=hamlet_content_cleaned,
        parent_id=None,
        original_span=(0, len(hamlet_content_cleaned)),
        hierarchy_level=0,
    )
    splitter = LexicalDensitySplitter(
        min_words=8,
        overlap=4,
        max_words=32,
        target_density=0.7,
        strategy="shortest",
    )

    chunks = splitter.split(document_chunk)

    docment = ChunkedDocument(
        filename="Hamlet.txt",
        chunks=chunks,
    )

    embedder =

    searcher = SimilaritySearcher(
        document=docment,
        operation_level=1,
        query_vector="to be or not to be",
    )