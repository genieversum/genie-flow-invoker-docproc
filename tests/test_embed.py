from datetime import time, datetime, timedelta
from functools import partial
from http import HTTPStatus

import requests

from invoker.docproc.embed import EmbeddingManager
from invoker.docproc.model.vectorizer import VectorResponse
from tests.conftest import MockRequestResponse


def test_embedding_mgr(monkeypatch):

    text = "to be or not to be"
    vector = [0.0, 0.1, 0.0]

    def embedding_response(*args, **kwargs):
        vector_response = VectorResponse(
            text=text,
            vector=vector,
            dim=3,
        )
        vector_response_json = vector_response.model_dump_json(indent=2)

        return MockRequestResponse(
            status_code=200,
            text=vector_response_json,
        )

    mgr = EmbeddingManager(
        text2vec_url="http://localhost:8000",
        pooling_strategy="mean",
    )

    monkeypatch.setattr(requests, "post", embedding_response)

    vector_response = mgr.make_embedding_request(text)

    assert vector_response == vector


def test_embedding_mgr_timeout(monkeypatch):
    text = "to be or not to be"
    vector = [0.0, 0.1, 0.0]

    times_when_called = []

    def embedding_response(called, *args, **kwargs):
        vector_response = VectorResponse(
            text=text,
            vector=vector,
            dim=3,
        )
        vector_response_json = vector_response.model_dump_json(indent=2)

        if len(called) < 3:
            called.append(datetime.now())
            return MockRequestResponse(status_code=HTTPStatus.REQUEST_TIMEOUT)

        return MockRequestResponse(
            status_code=HTTPStatus.OK,
            text=vector_response_json,
        )

    mgr = EmbeddingManager(
        text2vec_url="http://localhost:8000",
        pooling_strategy="mean",
    )

    monkeypatch.setattr(
        requests,
        "post",
        partial(embedding_response, called=times_when_called),
    )

    start = datetime.now()
    vector_response = mgr.make_embedding_request(text)
    end = datetime.now()

    assert (end - start) > timedelta(seconds=1)
    assert len(times_when_called) == 3
    assert vector_response == vector
