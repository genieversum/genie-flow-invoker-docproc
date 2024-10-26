from typing import Optional, NamedTuple

import numpy as np

from invoker.docproc.model import DocumentChunk

ChunkVector = NamedTuple(
    "ChunkVector",
    [
        ("chunk", DocumentChunk),
        ("vector", np.ndarray),
        ("distance", Optional[float]),
    ]
)
