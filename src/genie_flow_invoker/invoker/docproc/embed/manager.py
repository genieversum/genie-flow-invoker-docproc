from http import HTTPStatus

import requests
from loguru import logger

from genie_flow_invoker.invoker.docproc.backoff_caller import BackoffCaller
from genie_flow_invoker.invoker.docproc.model.vectorizer import (
    VectorInputConfig,
    VectorInput,
    VectorResponse,
)


def request_vector(url: str, in_vec: VectorInput) -> list[float]:
    input_dict = in_vec.model_dump()
    response = requests.post(url=url, json=input_dict)
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
    vector_response = VectorResponse.model_validate_json(response.text)
    return vector_response.vector


class EmbeddingManager:

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

    def make_embedding_request(self, text: str) -> list[float]:
        vector_input = VectorInput(text=text, config=self._vector_input_config)

        return self._backoff_caller.call(
            func=request_vector,
            url=f"{self._text2vec_url}/vectors",
            in_vec=vector_input,
        )
