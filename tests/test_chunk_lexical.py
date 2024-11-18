import nltk
from nltk import TreebankWordTokenizer

from genie_flow_invoker.invoker.docproc.chunk import (
    LexicalDensitySplitter,
    LexicalDensitySplitInvoker,
)
from genie_flow_invoker.invoker.docproc.model import ChunkedDocument, DocumentChunk


def test_lexical_chunk_no_meaning():
    chunker = LexicalDensitySplitter(5, 15, 2, 0.8)

    chunk = DocumentChunk(
        content="this is just a bunch of nonsense.",
        original_span=(0,1000),
        hierarchy_level=0,
        parent_id=None,
    )

    chunks = chunker.split(chunk)
    assert len(chunks) == 0


def test_lexical_chunk_with_meaning(sweden_text):
    chunker = LexicalDensitySplitter(15, 40, 5, 0.6)

    document = DocumentChunk(
        content=sweden_text,
        original_span=(0, len(sweden_text)),
        hierarchy_level=0,
        parent_id=None,
    )

    chunks = chunker.split(document)

    tokenizer = TreebankWordTokenizer()
    lexical_tags = {
        "NN",
        "NNS",
        "NNP",
        "NNPS",
        "JJ",
        "JJR",
        "JJS",
        "VB",
        "VBD",
        "VBG",
        "VBN",
        "VBP",
        "VBZ",
        "RB",
        "RBR",
        "RBS",
    }

    for c in chunks:
        words = tokenizer.tokenize(c.content)
        assert 15 <= len(words) <= 40

        pos: list[tuple[str, str]] = nltk.tag.pos_tag(words)
        lexical_words = sum(1 for p in pos if p[1] in lexical_tags)
        assert lexical_words / len(words) >= 0.6


def test_hierarchy_chunking(hamlet_path, hamlet_content_cleaned):
    hamlet_first_part = hamlet_content_cleaned[0:5000]

    input_document = ChunkedDocument(
        filename=hamlet_path,
        document_metadata=dict(),
        chunks=[
            DocumentChunk(
                content=hamlet_first_part,
                original_span=(0, len(hamlet_first_part)),
                hierarchy_level=0,
                parent_id=None,
            )
        ]
    )

    big_chunk_invoker = LexicalDensitySplitInvoker(
        min_words=500,
        max_words=2000,
        overlap=50,
        target_density=0.2,
        strategy="shortest",
    )

    big_chunk_result_json = big_chunk_invoker.invoke(input_document.model_dump_json())
    big_chunk_result = ChunkedDocument.model_validate_json(big_chunk_result_json)
    big_chunk_map = {
        d.chunk_id:d
        for d in big_chunk_result.chunks
    }

    nr_top_chunks = 0
    for chunk in big_chunk_result.chunks:
        assert nr_top_chunks <= 1

        if chunk.parent_id is not None:
            assert chunk.parent_id in big_chunk_map
            assert big_chunk_map[chunk.parent_id].hierarchy_level == chunk.hierarchy_level - 1
            assert hamlet_first_part[chunk.original_span[0]:chunk.original_span[1]] == chunk.content
        else:
            assert chunk.hierarchy_level == 0
            nr_top_chunks += 1


    child_chunk_invoker = LexicalDensitySplitInvoker(
        min_words=50,
        max_words=200,
        overlap=5,
        target_density=0.6,
        strategy="best",
        operation_level=1
    )
    child_chunk_result_json = child_chunk_invoker.invoke(big_chunk_result_json)
    child_chunk_result = ChunkedDocument.model_validate_json(child_chunk_result_json)
    child_chunk_map = {
        d.chunk_id:d
        for d in child_chunk_result.chunks
    }

    nr_top_chunks = 0
    for chunk in child_chunk_result.chunks:
        assert nr_top_chunks <= 1

        if chunk.parent_id is not None:
            assert chunk.parent_id in child_chunk_map
            assert child_chunk_map[chunk.parent_id].hierarchy_level == chunk.hierarchy_level - 1
            assert hamlet_first_part[chunk.original_span[0]:chunk.original_span[1]] == chunk.content
        else:
            assert chunk.hierarchy_level == 0
            nr_top_chunks += 1
