import sys
from collections import defaultdict
from typing import NamedTuple, Optional
from unicodedata import category

import nltk
from nltk import TreebankWordTokenizer, TreebankWordDetokenizer, PunktSentenceTokenizer

from loguru import logger

from genie_flow_invoker.invoker.docproc.chunk import AbstractSplitter
from genie_flow_invoker.doc_proc import DocumentChunk

WordSpanIndex = NamedTuple(
    "WordSpanIndex",
    [
        ("word", str),
        ("span", tuple[int, int]),
        ("word_index", int),
        ("sentence_index", int),
    ],
)

PUNCTUATION_CHARACTERS = {
    chr(i) for i in range(sys.maxunicode + 1) if category(chr(i)).startswith("P")
}


def _scan_till_sentence_break(
        idx: int,
        words_spans: list[WordSpanIndex],
        direction: int,
) -> int:
    """
    Scan forward or backward in the list of words_spans until a sentence break is found.
    If the scan goes beyond the list bounds, the appropriate boundary value is returned.
    If the start position is already at a sentence break, the scan will return that position.

    :param idx: The starting position to scan from.
    :param words_spans: The list of WordSpanIndex tuples.
    :param direction: The direction to scan in. 1 for forward, -1 for backward.
    :return: The index of the first sentence break found.
    """
    first_word_idx = 0
    last_word_idx = len(words_spans) - 1

    try:
        previous_sentence = words_spans[idx].sentence_index
        while words_spans[idx + direction].sentence_index == previous_sentence:
            idx += direction
        return idx
    except IndexError:
        return last_word_idx if direction > 0 else first_word_idx


class FixedWordsSplitter(AbstractSplitter):

    def __init__(
        self,
        max_words: int,
        overlap: int,
        ignore_stopwords: bool = False,
        drop_trailing_chunks: bool = False,
        break_on_punctuation: bool = False,
    ):
        """
        A Splitter that chunks a text into fixed word-count chunks. Every output chunk will
        have the same number of words, unless the flag `ignore_stopwords` is True. In that case,
        all stop words are ignored in the counting. (Currently only English stop words are regarded.)

        The splitter starts from the beginning of the content and strides through that, generating
        new chunks as it passes. This means that the final chunks will be smaller than the `max_words`
        parameter indicates. Unless the parameter `drop_trailing_chunks` is set to `True`, in which
        case final chunks that are smaller than the `max_words` will be dropped.

        The `max_words` argument determines the number of words that should be included in
        each resulting chunk.

        The `overlap` argument gives the number of words to skip forward for each consecutive
        chunk.

        If "break_on_punctuation" is set to True, the splitter will make sure that the chunks
        always contain full sentences. This means that each chunk will end with the last sentence
        that still fits the max_words constraint. If there is only one sentence, this chunker
        will then cross the max_words constraint and create a chunk with that sentence.

        In that case, the chunker will always start with a full sentence. When jumping ahead
        by `overlap` words, the chunker will skip to the first full sentence that follows.

        If "break_on_punctuation" is set to False, the splitter will chunk irrespective of the
        sentence boundaries.

        :param max_words: the maximum number of words that should be included into each chunk.
        :param overlap: the number of words to skip forward for each consecutive chunk.
        :param ignore_stopwords: whether to ignore stop words in the chunks.
        :param drop_trailing_chunks: whether to drop trailing chunks that are smaller than
            `max_words`.
        :param break_on_punctuation:
            whether to break on punctuation. If True, the splitter will make sure that the
            chunks are always full sentences. If False, the splitter will chunk irrespective
            of the sentence boundaries.
        """
        self._max_words = max_words
        self._overlap = overlap
        self._filter_words = (
            set(nltk.corpus.stopwords.words("english")) if ignore_stopwords else set()
        ).union(PUNCTUATION_CHARACTERS)
        self._drop_smaller_chunks = drop_trailing_chunks
        self._break_on_punctuation = break_on_punctuation

        self.sentence_splitter = PunktSentenceTokenizer()
        self.word_splitter = TreebankWordTokenizer()
        self.word_joiner = TreebankWordDetokenizer()


    def _determine_chunk_end(self, start: int, words_spans: list[WordSpanIndex]) -> int:
        """
        Determine the end index of the chunk, starting at the given `start` index.
        If breaking on punctuation is enabled, we skip backward to the end of the last full
        sentence, keeping the chunk within the `max_words` constraint. Unless there is only
        one sentence in the chunk, in which case we skip forward to the end of that sentence.

        :param start: The start index within the `words_spans` list.
        :param words_spans: A list of WordSpanIndex tuples.
        :return: The last index that should be included in the `words_spans` list.
        """
        start_sentence_index = words_spans[start].sentence_index
        last_word_idx = len(words_spans) - 1
        end = start + self._max_words - 1

        # we are at or passed the end of the list, return the last index
        if end > last_word_idx:
            return last_word_idx

        # no need to break on punctuation, return the end index
        if not self._break_on_punctuation:
            return end

        # if we end up in the same sentence, skip forward till we cross to the next sentence
        if words_spans[end].sentence_index == start_sentence_index:
            return _scan_till_sentence_break(end, words_spans, 1)

        # else, scan backwards to the end of the last full sentence
        return _scan_till_sentence_break(end, words_spans, -1) - 1

    def _determine_chunk_start(
            self,
            start: int,
            words_spans: list[WordSpanIndex],
    ) -> Optional[int]:
        """
        Determine the next start index for a chunk, starting at the given `start` index. If
        there is no next start index, return None. If breaking on punctuation is enabled,
        we skip forward to the first full sentence that starts after the current start
        plus the defined overlap.

        :param start: the start index within the `words_spans` list.
        :param words_spans: the list of WordSpanIndex tuples.
        :return: the optional next start index.
        """
        new_start = start + self._overlap
        last_word_idx = len(words_spans) - 1

        if new_start > last_word_idx:
            return None

        if not self._break_on_punctuation:
            return new_start

        new_start = _scan_till_sentence_break(new_start, words_spans, 1) + 1
        if new_start > last_word_idx:
            return None

        return new_start

    def _tokenize(self, content: str) -> list[WordSpanIndex]:
        """
        Tokenizes the provided content into words while keeping track of their positions
        within the original content and their respective sentences. Each word is annotated
        with its span, word index, and the sentence index to which it belongs.

        :param content: Content to be tokenized
        :type content: str
        :return: List of word span index objects that include word, span, word index, and
            sentence index information
        :rtype: list[WordSpanIndex]
        """
        sentences = self.sentence_splitter.tokenize(content)
        sentence_spans = list(self.sentence_splitter.span_tokenize(content))

        words_spans: list[WordSpanIndex] = []
        word_idx = 0
        for sentence_idx, sentence in enumerate(sentences):
            words = self.word_splitter.tokenize(sentence)
            word_spans = list(self.word_splitter.span_tokenize(sentence))
            span_origin = sentence_spans[sentence_idx][0]
            for word, span in zip(words, word_spans):
                words_spans.append(
                    WordSpanIndex(
                        word=word,
                        span=(span_origin + span[0], span_origin + span[1]),
                        word_index=word_idx,
                        sentence_index=sentence_idx,
                    )
                )
                word_idx += 1
        return words_spans

    def _detokenize(self, words_spans: list[WordSpanIndex]) -> str:
        """
        Detokenizes a list of word spans back into a text string. The method preserves the
        sentence structure by grouping words by their sentence index and then joining the
        sentences with newlines.
    
        :param words_spans: List of WordSpanIndex objects containing words and their metadata
        :type words_spans: list[WordSpanIndex]
        :return: The detokenized text with sentences joined by newlines
        :rtype: str
        """
        sentence_index: dict[int, list[WordSpanIndex]] = defaultdict(list)
        for word_span in words_spans:
            sentence_index[word_span.sentence_index].append(word_span)
        sentences: dict[int, str] = defaultdict(str)
        for sentence_idx, word_spans in sentence_index.items():
            sentences[sentence_idx] = self.word_joiner.detokenize(
                [word_span.word for word_span in word_spans]
            )
        return "\n".join(sentences.values())

    def _map_back_to_unfiltered(
            self,
            filtered_words_spans: list[WordSpanIndex],
            words_spans: list[WordSpanIndex],
            start_idx: int,
            end_idx: int,
    ):
        """
        Maps the chunk that is identified by the given start and end indices back to the
        original unfiltered words_spans list.

        If breaking on punctuation is enabled, add a possible punctuation mark at the end of
        the last word in the chunk, if it is present in the original unfiltered words_spans.

        :param filtered_words_spans: the filtered words_spans list
        :param words_spans: The original unfiltered words_spans list
        :param start_idx: The start index of the chunk
        :param end_idx: the end index of the chunk
        :return: a splice of the original unfiltered words_spans list that contains the chunk
        """
        first_word_idx = filtered_words_spans[start_idx].word_index
        last_word_idx = filtered_words_spans[end_idx].word_index
        try:
            if (
                    self._break_on_punctuation
                    and words_spans[last_word_idx + 1].word in PUNCTUATION_CHARACTERS
            ):
                return words_spans[first_word_idx : last_word_idx + 2]
        except IndexError:
            pass

        return words_spans[first_word_idx : last_word_idx + 1]

    def split(self, document: DocumentChunk) -> list[DocumentChunk]:
        words_spans = self._tokenize(document.content)

        filtered_words_spans: list[WordSpanIndex] = [
            word_span
            for word_span in words_spans
            if word_span.word not in self._filter_words
        ]

        chunks: list[DocumentChunk] = []
        chunk_start: Optional[int] = 0
        first_smaller_chunk_seen = False
        while chunk_start is not None and chunk_start < len(filtered_words_spans):
            chunk_end = self._determine_chunk_end(chunk_start, filtered_words_spans)

            chunk_word_spans = self._map_back_to_unfiltered(
                filtered_words_spans,
                words_spans,
                chunk_start,
                chunk_end,
            )
            chunk_content = self._detokenize(chunk_word_spans)
            chunk_start = self._determine_chunk_start(chunk_start, filtered_words_spans)

            if self._drop_smaller_chunks and len(chunk_word_spans) < self._max_words:
                if first_smaller_chunk_seen:
                    logger.debug(
                        "ignoring chunk because it is smaller than the max words: '{chunk_content}'",
                        chunk_content=chunk_content,
                    )
                    continue
                else:
                    first_smaller_chunk_seen = True

            logger.debug(
                "chunked chunk {parent_id} into words: '{chunk_content}'",
                parent_id=document.chunk_id,
                chunk_content=chunk_content,
            )

            chunks.append(
                DocumentChunk(
                    parent_id=document.chunk_id,
                    content=chunk_content,
                    original_span=(
                        chunk_word_spans[0].span[0],
                        chunk_word_spans[-1].span[1],
                    ),
                    hierarchy_level=document.hierarchy_level + 1,
                )
            )

        return chunks
