from genie_flow_invoker import GenieInvoker

from invoker.docproc.chunk.lexical_density import LexicalDensitySplitter
from invoker.docproc.codec import PydanticInputDecoder, PydanticOutputEncoder
from invoker.docproc.model import ParsedDocument, ChunkedDocument


class LexicalDensitySplitInvoker(
    GenieInvoker,
    PydanticInputDecoder[ParsedDocument],
    PydanticOutputEncoder[ChunkedDocument],
):

    def __init__(
            self,
            min_words: int,
            max_words: int,
            overlap: int,
            target_density: float,
    ):
        self.splitter = LexicalDensitySplitter(
            min_words=min_words,
            max_words=max_words,
            overlap=overlap,
            target_density=target_density,
        )

    @classmethod
    def from_config(cls, config: dict):
        """
        Create a new LexicalDensitySplitInvoker instance from configuration. The
        configuration is expected to have the following keys:
        - min_words: (int, default 5) the minimal number ofo words in a chunk
        - max_words: (int, default 15) the maximal number ofo words in a chunk
        - overlap: (int, default 2) the overlap between chunks
        - target_density: (float, default 0.8) the target density of the chunk
        :param config: the configuration as ready from the meta.yaml file
        :return: a new LexicalDensitySplitInvoker instance
        """
        min_words = config.get("min_words", 5)
        max_words = config.get("max_words", 15)
        overlap = config.get("overlap", 2)
        target_density = config.get("target_density", 0.8)
        return cls(min_words, max_words, overlap, target_density)

    def invoke(self, content: str) -> str:
        document = self._decode_input(content)
        chunks = self.splitter.split(document)
        chunked_document = ChunkedDocument(
            filename=document.filename,
            chunks=chunks,
        )
        return self._encode_output(chunked_document)
