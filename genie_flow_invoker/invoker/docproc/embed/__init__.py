from genie_flow_invoker import GenieInvoker
from genie_flow_invoker.utils import get_config_value

from invoker.docproc.codec import PydanticInputDecoder, PydanticOutputEncoder
from invoker.docproc.embed.manager import EmbeddingManager
from invoker.docproc.model import ChunkedDocument


class EmbedInvoker(
    GenieInvoker,
    PydanticInputDecoder[ChunkedDocument],
    PydanticOutputEncoder[ChunkedDocument],
):

    def __init__(
        self,
        text2vec_url: str,
        pooling_strategy: str,
        backoff_max_time=61,
        backoff_max_tries=10,
    ):
        self._embedding_manager = EmbeddingManager(
            text2vec_url=text2vec_url,
            pooling_strategy=pooling_strategy,
            backoff_max_time=backoff_max_time,
            backoff_max_tries=backoff_max_tries,
        )

    @classmethod
    def from_config(cls, config: dict):
        text2vec_url = get_config_value(
            config,
            "TEXT_2_VEC_URL",
            "text2vec_url",
            "Text2Vec URL",
        )
        pooling_strategy = get_config_value(
            config,
            "POOLING_STRATEGY",
            "pooling_strategy",
            "Pooling Strategy",
            None,
        )
        backoff_max_time = get_config_value(
            config,
            "VECTORIZER_BACKOFF_MAX_TIME",
            "backoff_max_time",
            "Max backoff time (seconds)",
            61,
        )
        backoff_max_tries = get_config_value(
            config,
            "VECTORIZER_MAX_BACKOFF_TRIES",
            "backoff_max_tries",
            "Max backoff tries",
            15,
        )

        return cls(
            text2vec_url,
            pooling_strategy,
            backoff_max_time,
            backoff_max_tries,
        )

    def invoke(self, content: str) -> str:
        chunked_document = self._decode_input(content)

        for chunk in chunked_document.chunks:
            vector = self._embedding_manager.make_embedding_request(chunk.content)
            chunk.embedding = vector

        return self._encode_output(chunked_document)
