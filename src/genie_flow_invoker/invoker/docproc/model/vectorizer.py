from typing import Optional

from pydantic import BaseModel, Field


class VectorInputConfig(BaseModel):
    pooling_strategy: str = Field(
        description="The pooling strategy to use for pooling vectors",
    )


class VectorInput(BaseModel):
    text: str = Field(
        description="The text that needs to be embedded into a vector",
    )
    config: Optional[VectorInputConfig] = Field(
        None,
        description="The configuration for the embedder",
    )


class VectorResponse(BaseModel):
    text: str = Field(
        description="The text that has been embedded into a vector",
    )
    vector: list[float] = Field(
        description="The embedding of the text",
    )
    dim: int = Field(
        description="The dimension of the vector",
    )
