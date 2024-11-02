from typing import Optional

import numpy as np
from genie_flow_invoker import GenieInvoker

from invoker.docproc.codec import PydanticInputDecoder, PydanticOutputEncoder
from invoker.docproc.model import SimilaritySearch, SimilarityResults
from invoker.docproc.similarity_search.search import SimilaritySearcher


class SimilaritySearchInvoker(
    GenieInvoker,
    PydanticInputDecoder[SimilaritySearch],
    PydanticOutputEncoder[SimilarityResults],
):

    def __init__(
        self,
        operation_level: Optional[int] = None,
        horizon: Optional[int] = None,
        top: Optional[int] = None,
        parent_strategy: Optional[str] = None,
        method: str = "cosine",
    ):
        self.operation_level = operation_level
        self.horizon = horizon
        self.top = top
        self.parent_strategy = parent_strategy
        self.method = method

    @classmethod
    def from_config(cls, config: dict):
        operation_level = config.get("operation_level", None)
        horizon = config.get("horizon", None)
        top = config.get("top", None)
        parent_strategy = config.get("parent_strategy", None)
        method = config.get("method", "cosine")
        return cls(
            operation_level=operation_level,
            horizon=horizon,
            top=top,
            parent_strategy=parent_strategy,
            method=method,
        )

    def invoke(self, content: str) -> str:
        search_query = self._decode_input(content)
        for attribute in [
            "operation_level",
            "horizon",
            "top",
            "parent_strategy",
            "method",
        ]:
            if getattr(search_query, attribute) is None:
                setattr(search_query, attribute, getattr(self, attribute))

        similarity_search = SimilaritySearcher(
            chunks=search_query.chunks,
            operation_level=search_query.operation_level,
            parent_strategy=search_query.parent_strategy,
        )
        similarities = similarity_search.calculate_similarities(
            query_vector=np.array(search_query.query_embedding),
            horizon=search_query.horizon,
            top=search_query.top,
            method=search_query.method,
        )

        result_document = SimilarityResults(
            filename=search_query.filename,
            chunk_distances=similarities,
        )

        return self._encode_output(result_document)
