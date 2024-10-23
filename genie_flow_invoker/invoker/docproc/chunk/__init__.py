from abc import ABC
from typing import Optional, Iterator, Literal

from genie_flow_invoker import GenieInvoker

from invoker.docproc.chunk.lexical_density import LexicalDensitySplitter, LexicalSplitStrategyType
from invoker.docproc.chunk.splitter import AbstractSplitter
from invoker.docproc.codec import PydanticInputDecoder, PydanticOutputEncoder
from invoker.docproc.model import ChunkedDocument, DocumentChunk


class AbstractSplitterInvoker(
    GenieInvoker,
    PydanticInputDecoder[ChunkedDocument],
    PydanticOutputEncoder[ChunkedDocument],
    ABC,
):
    """
    A Splitter Invoker takes in a document and chunks it up into smaller chunks.

    The input is a ChunkedDocument. That Chunked Document should include the chunks
    that need to be split into smaller chunks. When this invoker runs, it returns the
    same ChunkedDocument as the input ChunkedDocument, except that new chunks that are
    created, are added to the list of chunks, each with:
     - an increased hierarchy_level
     - a reference to the parent_id of the chunk that they were created from
    """

    def __init__(
            self,
            operation_level: Optional[int] = None,
    ):
        """
        Create a new instance.

        :param operation_level: an optional level at which this invoker will operate.
        If not given, all chunks will be split into smaller chunks. If given, only
        chunks of that level will be split.
        """
        self._operation_level = operation_level

    def get_splitter(self) -> AbstractSplitter:
        raise NotImplementedError("Subclasses must override this property")

    def chunk_iterator(self, chunks: list[DocumentChunk]) -> Iterator[DocumentChunk]:
        for chunk in chunks:
            if (
                    self._operation_level is None or
                    chunk.hierarchy_level == self._operation_level
            ):
                yield chunk

    def invoke(self, content: str) -> str:
        document = self._decode_input(content)

        new_chunks = []
        for chunk in self.chunk_iterator(document.chunks):
            new_chunks.extend(self.get_splitter().split(chunk))
        document.chunks.extend(new_chunks)

        return self._encode_output(document)


class LexicalDensitySplitInvoker(AbstractSplitterInvoker):
    """
    This split invoker uses Lexical Density to determine the split.
    """

    def __init__(
            self,
            min_words: int,
            max_words: int,
            overlap: int,
            target_density: float,
            strategy: LexicalSplitStrategyType,
            operation_level: Optional[int] = None,
    ):
        super().__init__(operation_level)
        self._splitter = LexicalDensitySplitter(
            min_words=min_words,
            max_words=max_words,
            overlap=overlap,
            strategy=strategy,
            target_density=target_density,
        )

    def get_splitter(self) -> AbstractSplitter:
        return self._splitter

    @classmethod
    def from_config(cls, config: dict):
        """
        Create a new LexicalDensitySplitInvoker instance from configuration. The
        configuration is expected to have the following keys:
        - min_words: (int, default 5) the minimal number ofo words in a chunk
        - max_words: (int, default 15) the maximal number ofo words in a chunk
        - overlap: (int, default 2) the overlap between chunks
        - target_density: (float, default 0.8) the target density of the chunk
        - operation_level: (int, default None) the hierarchy level that should be split
        :param config: the configuration as ready from the meta.yaml file
        :return: a new LexicalDensitySplitInvoker instance
        """
        min_words = config.get("min_words", 5)
        max_words = config.get("max_words", 15)
        overlap = config.get("overlap", 2)
        target_density = config.get("target_density", 0.8)
        strategy = config.get("strategy", "shortest")
        operation_level = config.get("operation_level", None)
        return cls(min_words, max_words, overlap, target_density, strategy, operation_level)
