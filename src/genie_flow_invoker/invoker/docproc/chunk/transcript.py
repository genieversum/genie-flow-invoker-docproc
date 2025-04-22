from io import StringIO

import webvtt
from webvtt import Caption
from webvtt.errors import MalformedFileError
from nltk import PunktSentenceTokenizer, TreebankWordTokenizer, TreebankWordDetokenizer

from genie_flow_invoker.invoker.docproc.chunk import AbstractSplitter
from genie_flow_invoker.doc_proc import DocumentChunk
from loguru import logger


_UNKNOWN_PARTY_NAME = "UNKNOWN"


def _create_chunk(caption: Caption, parent: DocumentChunk) -> DocumentChunk:
    return DocumentChunk(
        parent_id=parent.chunk_id,
        hierarchy_level=parent.hierarchy_level + 1,
        content=caption.text,
        original_span=(caption.start_in_seconds, caption.end_in_seconds),
        custom_properties={
            "party_name": caption.voice or _UNKNOWN_PARTY_NAME,
            "seconds_start": caption.start_in_seconds,
            "seconds_end": caption.end_in_seconds,
            "duration": caption.end_in_seconds - caption.start_in_seconds,
            "identifier": caption.identifier,
        }
    )


def _merge_consecutive_captions(captions: list[Caption]) -> list[Caption]:
    """
    Return a list of captions where consecutive captions from the same voice are merged.
    Text is merged by adding these, separated by a newline character.
    The time span is adapted such that each caption has the end time of the
    last caption in one consecutive block for one voice.

    :param captions: a list of captions to merge.
    :return: a list of merged captions
    """
    merged_captions: list[Caption] = list()
    for caption in captions:
        if len(merged_captions) == 0:
            merged_captions.append(caption)
        elif (
                caption.voice is not None
                and caption.voice == merged_captions[-1].voice
        ):
            merged_captions[-1].lines.append(caption.text)
            merged_captions[-1].end = caption.end
        else:
            merged_captions.append(caption)
    return merged_captions


class BrokenSentencesCleaner:

    def __init__(self):
        """
        This cleaner takes in a text on which sentences may be broken up by newlines.
        We tokenize by sentences, then by words and then reconstruct the text.
        In the output text, every sentence is separated by a single newline character.
        """
        self.sent_tokenizer = PunktSentenceTokenizer()
        self.word_tokenizer = TreebankWordTokenizer()
        self.detokenizer = TreebankWordDetokenizer()

    def clean(self, text: str) -> str:
        caption_sentences = self.sent_tokenizer.tokenize(text)
        caption_words = [
            self.word_tokenizer.tokenize(sentence)
            for sentence in caption_sentences
        ]
        return "\n".join(
            self.detokenizer.detokenize(words)
            for words in caption_words
        )


class TranscriptSplitter(AbstractSplitter):

    def split(self, parent_chunk: DocumentChunk) -> list[DocumentChunk]:
        try:
            document_stream = StringIO(parent_chunk.content)
            vtt_captions = webvtt.read_buffer(document_stream)
        except MalformedFileError:
            logger.debug(
                "Could not parse a document chunk as WebVTT, "
                "starting with '{doc_start}' and ending with '{doc_end}' ",
                doc_start=parent_chunk.content[:100],
                doc_end=parent_chunk.content[-100:],
            )
            logger.warning("Could not parse document as WebVTT.")
            return []

        vtt_captions = _merge_consecutive_captions(vtt_captions.captions)
        cleaner = BrokenSentencesCleaner()

        chunks: list[DocumentChunk] = list()
        for caption in vtt_captions:
            caption.text = cleaner.clean(caption.text)
            chunks.append(_create_chunk(caption, parent_chunk))

        return chunks
