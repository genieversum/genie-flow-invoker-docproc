import math
import re

import nltk
from nltk import TreebankWordTokenizer

from genie_flow_invoker.invoker.docproc.chunk.word_splitter import FixedWordsSplitter, \
    PUNCTUATION_CHARACTERS, _scan_till_sentence_break
from genie_flow_invoker.doc_proc import DocumentChunk


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
        max_words=5,
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
    nr_words = sum(1 for word in words if word not in PUNCTUATION_CHARACTERS)

    chunks = splitter.split(document)

    nr_chunks = len(chunks)
    assert nr_chunks == int(math.ceil(nr_words / 5))

    previous_length = None
    trailing_off = False
    for chunk in chunks:
        chunk_words = word_splitter.tokenize(chunk.content)
        chunk_length = sum(1 for w in chunk_words if w not in PUNCTUATION_CHARACTERS)

        if previous_length is None:
            previous_length = chunk_length
            assert previous_length == min(15, nr_words)
            continue

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


def test_scan_for_sentence_break():
    content = (
        "This is a sentence. "  # idx 0 1 2 3 4
        "This is another sentence. "  # idx 5 6 7 8 9
        "And, finally, this is a third sentence."  # idx 10 11 12 13 14 15 16 17 18 19
    )
    splitter = FixedWordsSplitter(
        max_words=5,
        overlap=2,
    )

    word_spans = splitter._tokenize(content)

    start_second_sentence = _scan_till_sentence_break(0, word_spans, 1)
    assert start_second_sentence == 4

    start_third_sentence = _scan_till_sentence_break(len(word_spans) - 1, word_spans, -1)
    assert start_third_sentence == 10


def test_chunk_word_punctuation(sweden_text):
    document = DocumentChunk(
        original_span=(0, len(sweden_text)),
        hierarchy_level=0,
        content=sweden_text,
        parent_id=None,
    )
    splitter = FixedWordsSplitter(
        max_words=15,
        overlap=5,
        ignore_stopwords=False,
        drop_trailing_chunks=False,
        break_on_punctuation=True,
    )

    chunks = splitter.split(document)

    assert chunks[0].content.startswith("Sweden")
    assert chunks[0].content.endswith("Northern Europe.")
    assert len(chunks[0].content.split(" ")) > 15

    assert chunks[1].content.startswith("It borders")
    assert len(chunks[1].content.split(" ")) < 15

    for chunk in chunks:
        assert re.match("[A-Z]", chunk.content[0]) is not None
        assert chunk.content.endswith(".")
