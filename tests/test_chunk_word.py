import math

import nltk
from nltk import TreebankWordTokenizer

from genie_flow_invoker.invoker.docproc.chunk.word_splitter import FixedWordsSplitter, PUNCTUATION_CHARACTERS
from genie_flow_invoker.invoker.docproc.model import DocumentChunk


def test_chunk_empty():
    document = DocumentChunk(
        original_span=(0, 0),
        hierarchy_level=0,
        content="",
        parent_id=None,
    )

    splitter = FixedWordsSplitter(
        max_words=15,
        overlap=5,
        ignore_stopwords=False,
    )

    chunks = splitter.split(document)

    assert len(chunks) == 0


def test_exact_overlap_count():
    content = "one two three four five"
    document = DocumentChunk(
        original_span=(0, len(content)),
        hierarchy_level=0,
        content=content,
        parent_id=None,
    )
    splitter = FixedWordsSplitter(
        max_words=15,
        overlap=5,
    )

    chunks = splitter.split(document)

    assert len(chunks) == 1


def test_overlap_plus_one():
    content = "one two three four five six"
    document = DocumentChunk(
        original_span=(0, len(content)),
        hierarchy_level=0,
        content=content,
        parent_id=None,
    )
    splitter = FixedWordsSplitter(
        max_words=15,
        overlap=5,
    )

    chunks = splitter.split(document)

    assert len(chunks) == 2
    assert chunks[0].content == content
    assert chunks[1].content == "six"


def test_chunk_word_simple(netherlands_text):
    document = DocumentChunk(
        original_span=(0, len(netherlands_text)),
        hierarchy_level=0,
        content=netherlands_text,
        parent_id=None,
    )
    splitter = FixedWordsSplitter(
        max_words=15,
        overlap=5,
        ignore_stopwords=False,
    )

    word_splitter = TreebankWordTokenizer()
    words = word_splitter.tokenize(netherlands_text)
    spans = word_splitter.span_tokenize(netherlands_text)

    punctuation_indices = {i for i in range(len(words)) if words[i] in PUNCTUATION_CHARACTERS}
    words = [w for i, w in enumerate(words) if i not in punctuation_indices]
    spans = [s for i, s in enumerate(spans) if i not in punctuation_indices]

    span_word_start_map = {
        spans[i][0]:i
        for i in range(len(words))
    }
    span_word_end_map = {
        spans[i][1]:i
        for i in range(len(words))
    }

    chunks = splitter.split(document)

    nr_chunks = len(chunks)
    expected_nr_chunks = int(math.ceil(len(words) / 5))
    assert nr_chunks == expected_nr_chunks

    previous_length = None
    trailing_off = False
    for i, chunk in enumerate(chunks):
        first_word_index = span_word_start_map[chunk.original_span[0]]
        last_word_index = span_word_end_map[chunk.original_span[1]]
        chunk_length = last_word_index - first_word_index + 1

        if i == 0:
            assert chunk_length == min(15, len(words))
        else:
            if not trailing_off:
                assert chunk_length <= previous_length
            else:
                assert chunk_length < previous_length
            trailing_off = chunk_length < previous_length

        previous_length = chunk_length


def test_chunk_no_trailing_off():
    content = "one two three four five six"
    document = DocumentChunk(
        original_span=(0, len(content)),
        hierarchy_level=0,
        content=content,
        parent_id=None,
    )
    splitter = FixedWordsSplitter(
        max_words=15,
        overlap=5,
        drop_trailing_chunks=True,
    )
    chunks = splitter.split(document)

    assert len(chunks) == 1


def test_drop_stopwords():
    # "this", "is", "a", "it" and "not" are stopwords
    content = "this is a carefully not constructed sentence is it not"
    document = DocumentChunk(
        original_span=(0, len(content)),
        hierarchy_level=0,
        content=content,
        parent_id=None,
    )
    splitter = FixedWordsSplitter(
        max_words=5,
        overlap=2,
        ignore_stopwords=True,
        drop_trailing_chunks=False,
    )
    chunks = splitter.split(document)

    assert len(chunks) == 2

    for chunk in chunks:
        words = chunk.content.split(" ")
        assert words[0] not in nltk.corpus.stopwords.words("english")
        assert words[-1] not in nltk.corpus.stopwords.words("english")
