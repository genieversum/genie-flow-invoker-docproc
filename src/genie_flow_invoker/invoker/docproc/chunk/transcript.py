from io import StringIO

import webvtt
from genie_flow_invoker.invoker.docproc.chunk import AbstractSplitter
from genie_flow_invoker.doc_proc import DocumentChunk
from loguru import logger
from webvtt import Caption
from webvtt.errors import MalformedFileError


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


def _extend_chunk(chunk: DocumentChunk, caption: Caption):
    chunk.content += "\n" + caption.text
    chunk.original_span = (chunk.original_span[0], caption.end_in_seconds)
    chunk.custom_properties["seconds_end"] = caption.end_in_seconds
    chunk.custom_properties["duration"] = \
            chunk.custom_properties["seconds_end"] - chunk.custom_properties["seconds_start"]
    chunk.custom_properties["identifier"] += "," + caption.identifier


class TranscriptSplitter(AbstractSplitter):

    def split(self, parent_chunk: DocumentChunk) -> list[DocumentChunk]:
        try:
            document_stream = StringIO(parent_chunk.content)
            captions = webvtt.read_buffer(document_stream)
        except MalformedFileError:
            logger.debug(
                "Could not parse a document chunk as WebVTT, "
                "starting with '{doc_start}' and ending with '{doc_end}' ",
                doc_start=parent_chunk.content[:100],
                doc_end=parent_chunk.content[-100:],
            )
            logger.warning("Could not parse document as WebVTT.")
            return []

        chunks: list[DocumentChunk] = list()
        for caption in captions:
            if (
                len(chunks) == 0
                or caption.voice is None
                or chunks[-1].custom_properties["party_name"] == _UNKNOWN_PARTY_NAME
                or caption.voice != chunks[-1].custom_properties["party_name"]
            ):
                chunks.append(_create_chunk(caption, parent_chunk))
            else:
                _extend_chunk(chunks[-1], caption)

        return chunks
