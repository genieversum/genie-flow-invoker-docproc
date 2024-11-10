import sys
from typing import NamedTuple
from unicodedata import category

import nltk
from nltk import TreebankWordTokenizer, TreebankWordDetokenizer

from loguru import logger

from genie_flow_invoker.invoker.docproc.chunk import AbstractSplitter
from genie_flow_invoker.invoker.docproc.model import DocumentChunk


WordSpanIndex = NamedTuple(
    "WordSpanIndex",
    [
        ("word", str),
        ("span", tuple[int, int]),
        ("word_index", int),
    ],
)

PUNCTUATION_CHARACTERS = {
    chr(i) for i in range(sys.maxunicode + 1) if category(chr(i)).startswith("P")
}


class FixedWordsSplitter(AbstractSplitter):
    """
    A Splitter that chunks a text into fixed word-count chunks. Every output chunk will
    have the same number of words, unless the flag `ignore_stopwords` is True. In that case,
    all stop words are ignored in the counting. (Currently only English stop words are regarded.)

    The splitter starts from the beginning of the content and strides through that, generating
    new chunks as it passes. This means that the final chunks will be smaller than the `max_words`
    parameter indicates. Unless the parameter `drop_trailing_chunks` is set to `True`, in which
    case final chunks that are smaller than the `max_words` will be dropped.
    """

    def __init__(
        self,
        max_words: int,
        overlap: int,
        ignore_stopwords: bool = False,
        drop_trailing_chunks: bool = False,
    ):
        """
        Create a new FixedWordsSplitter instance. The `max_words` argument determines the number
        of words that should be included into each resulting chunk.

        The `overlap` argument gives the number of words to skip forward for each consecutive
        chunk.

        :param max_words: the maximum number of words that should be included into each chunk.
        :param overlap: the number of words to skip forward for each consecutive chunk.
        :param ignore_stopwords: whether to ignore stop words in the chunks.
        :param drop_trailing_chunks: whether to drop trailing chunks that are smaller than
            `max_words`.
        """
        self._max_words = max_words
        self._overlap = overlap
        self._filter_words = (
            set(nltk.corpus.stopwords.words("english")) if ignore_stopwords else set()
        ).union(PUNCTUATION_CHARACTERS)
        self._drop_smaller_chunks = drop_trailing_chunks

    def split(self, document: DocumentChunk) -> list[DocumentChunk]:
        word_splitter = TreebankWordTokenizer()
        word_joiner = TreebankWordDetokenizer()

        span_origin = document.original_span[0]

        words = word_splitter.tokenize(document.content)
        spans = word_splitter.span_tokenize(document.content)
        words_spans = [
            WordSpanIndex(
                word=word,
                span=(span_origin + span[0], span_origin + span[1]),
                word_index=i,
            )
            for i, word, span in zip(range(len(words)), words, spans)
        ]

        filtered_words_spans: list[WordSpanIndex] = [
            word_span
            for word_span in words_spans
            if word_span.word not in self._filter_words
        ]

        chunks: list[DocumentChunk] = []
        start = 0
        previous_chunk_length = self._max_words
        while start < len(filtered_words_spans):
            chunk_length = min(self._max_words, len(filtered_words_spans) - start)
            if (
                self._drop_smaller_chunks
                and previous_chunk_length < self._max_words
                and chunk_length < self._max_words
            ):
                break

            filtered_chunk = filtered_words_spans[start : start + chunk_length]
            chunk_word_spans = words_spans[
                filtered_chunk[0].word_index : filtered_chunk[-1].word_index + 1
            ]
            chunk_words = [word_span.word for word_span in chunk_word_spans]
            logger.debug(chunk_words)

            words_chunk = DocumentChunk(
                parent_id=document.chunk_id,
                content=word_joiner.detokenize(chunk_words),
                original_span=(
                    chunk_word_spans[0].span[0],
                    chunk_word_spans[-1].span[1],
                ),
                hierarchy_level=document.hierarchy_level + 1,
            )
            chunks.append(words_chunk)
            start += self._overlap
            previous_chunk_length = chunk_length

        return chunks
