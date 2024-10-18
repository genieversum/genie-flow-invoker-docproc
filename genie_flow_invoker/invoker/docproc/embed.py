from http import HTTPStatus

import requests
from genie_flow_invoker import GenieInvoker
from genie_flow_invoker.utils import get_config_value
from loguru import logger

from invoker.docproc.backoff_caller import BackoffCaller
from invoker.docproc.codec import PydanticInputDecoder, PydanticOutputEncoder
from invoker.docproc.model import ChunkedDocument, EmbeddedChunkedDocument
from invoker.docproc.model.vectorizer import VectorInputConfig, VectorInput, VectorResponse


class EmbedInvoker(
    GenieInvoker,
    PydanticInputDecoder[ChunkedDocument],
    PydanticOutputEncoder[EmbeddedChunkedDocument]
):

    def __init__(
            self,
            text2vec_url: str,
            pooling_strategy: str,
            backoff_max_time=61,
            backoff_max_tries=10,
    ):
        self._text2vec_url = text2vec_url
        self._vector_input_config = VectorInputConfig(pooling_strategy=pooling_strategy)
        self._backoff_caller = BackoffCaller(
            TimeoutError,
            self.__class__,
            backoff_max_time,
            backoff_max_tries,
        )

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

    def _make_embedding_request(self, chunk: str) -> list[float]:

        def request_vector(url: str, in_vec: VectorInput) -> list[float]:
            input_json = in_vec.model_dump_json()
            response = requests.post(url=url, json=input_json)
            if response.status_code in [
                HTTPStatus.REQUEST_TIMEOUT,
                HTTPStatus.TOO_MANY_REQUESTS,
                HTTPStatus.INTERNAL_SERVER_ERROR,
            ]:
                logger.warning(
                    "Received status code {}, from embedding request. Raising a Timeout",
                    response.status_code,
                )
                raise TimeoutError()
            response.raise_for_status()
            vector_response = VectorResponse.model_validate_json(response.json())
            return vector_response.vector

        vector_input = VectorInput(text=chunk, config=self._vector_input_config)

        return self._backoff_caller.call(
            func=request_vector,
            url=f"{self._text2vec_url}/vectors",
            model_json=vector_input,
        )

    def invoke(self, content: str) -> str:
        chunked_document = self._decode_input(content)

        vectors = [
            self._make_embedding_request(d.document_chunk)
            for d in chunked_document.chunks
        ]

        result = EmbeddedChunkedDocument(
            filename=chunked_document.filename,
            chunks=chunked_document.chunks,
            embeddings=vectors,
        )

        return self._encode_output(result)
