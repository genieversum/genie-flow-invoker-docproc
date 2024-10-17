import backoff
import requests
from loguru import logger

from genie import GenieInvoker
from invoker.docproc.codec import PydanticInputDecoder, PydanticOutputEncoder
from invoker.docproc.model import ChunkedDocument, EmbeddedChunkedDocument, VectorResponse, \
    VectorInput, VectorInputConfig
from utils import get_config_value


class EmbedInvoker(
    GenieInvoker,
    PydanticInputDecoder[ChunkedDocument],
    PydanticOutputEncoder[EmbeddedChunkedDocument]
):

    def __init__(
            self, text2vec_url: str, pooling_strategy: str):
        self._text2vec_url = text2vec_url
        self._pooling_strategy = pooling_strategy
        max_value = self._backoff_max_time,
        max_tries = self._backoff_max_tries,

    @classmethod
    def from_config(cls, config: dict):
        text2vec_url = get_config_value(
            config,
            "TEXT_2_VEC_URL",
            'text2vec_url',
            "Text2Vec URL",
        )
        pooling_strategy = get_config_value(
            config,
            "POOLING_STRATEGY",
            'pooling_strategy',
            "Pooling Strategy",
            None,
        )
        return cls(text2vec_url, pooling_strategy)

    def _make_embedding_request(self, chunk: str) -> list[float]:
        vector_input = VectorInput(
            text=chunk,
            config=(
                VectorInputConfig(pooling_strategy=self._pooling_strategy)
                if self._pooling_strategy is not None
                else None
            )
        )

        def backoff_logger(details):
            logger.info(
                "Backing off {wait:0.1f} seconds after {tries} tries ",
                "for a {cls} invocation",
                **details,
                cls=self.__class__.__name__,
            )

        @backoff.on_exception(
            wait_gen=backoff.fibo,
            max_value=self._backoff_max_time,
            max_tries=self._backoff_max_tries,
            exception=TimeoutError,
            on_backoff=backoff_logger,
        )
        def _request_with_backoff():
            resp = requests.post(
                f"{self._text2vec_url}/vectors",
                json=vector_input.model_dump_json(),
            )
            resp.raise_for_status()
            return resp

        response = _request_with_backoff()
        vector_response = VectorResponse.model_validate_json(response.json())
        return vector_response.vector

    def invoke(self, content: str) -> str:
        chunked_document = self._decode_input(content)

        vectors = [
            self._make_embedding_request(d.document_chunk)
            for d in chunked_document.chunks
        ]

        result = EmbeddedChunkedDocument(
            **(chunked_document.model_dump()),
            embeddings=vectors,
        )

        return self._encode_output(result)
