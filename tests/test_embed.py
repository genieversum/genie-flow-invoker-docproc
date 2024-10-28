from datetime import time, datetime, timedelta
from functools import partial
from http import HTTPStatus

import requests

from invoker.docproc.embed import EmbeddingManager, EmbedInvoker
from invoker.docproc.model import ChunkedDocument, DocumentChunk
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


def test_embed_invoker(monkeypatch):
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

    monkeypatch.setattr(requests, "post", embedding_response)

    chunked_document = ChunkedDocument(
        filename="Hamlet.txt",
        document_metadata=dict(to_be_or_not_to_be=True),
        chunks=[
            DocumentChunk(
                content=text,
                original_span=(0, len(text)),
                parent_id=None,
                hierarchy_level=0,
            )
        ]
    )
    chunked_document_json = chunked_document.model_dump_json(indent=2)

    embedding_invoker = EmbedInvoker(
        text2vec_url="http://localhost:8000",
        pooling_strategy="mean",
    )

    embedded_document_json = embedding_invoker.invoke(chunked_document_json)
    embedded_document = ChunkedDocument.model_validate_json(embedded_document_json)

    assert embedded_document.filename == "Hamlet.txt"
    assert embedded_document.document_metadata["to_be_or_not_to_be"] == True
    assert len(embedded_document.chunks) == 1
    assert embedded_document.chunks[0].content == text
    assert embedded_document.chunks[0].original_span == (0, len(text))
    assert embedded_document.chunks[0].hierarchy_level == 0
    assert embedded_document.chunks[0].embedding == vector
