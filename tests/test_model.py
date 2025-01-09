from math import floor

import pytest

from genie_flow_invoker.invoker.docproc.model import ChunkedDocument, DocumentChunk

@pytest.fixture
def multilayered_chunked_document():

    def create_kids(parent: DocumentChunk, max_level: int) -> list[DocumentChunk]:
        if parent.hierarchy_level >= max_level:
            return []
        step = floor((parent.original_span[1] - parent.original_span[0]) / 5)
        kids_hierarchy_level = parent.hierarchy_level + 1
        kids = [
            DocumentChunk(
                content="some content",
                original_span=(
                    parent.original_span[0] + i * step,
                    min(
                        parent.original_span[1],
                        parent.original_span[0] + (i+1) * step,
                    )
                ),
                hierarchy_level=kids_hierarchy_level,
                parent_id=parent.chunk_id,
            )
            for i, kid_id in enumerate(
                range(
                    parent.original_span[0],
                    parent.original_span[1],
                    step
                )
            )
        ]
        grand_kids = []
        for kid in kids:
            grand_kids.extend(create_kids(kid, max_level))
        return kids + grand_kids

    grant_parent = DocumentChunk(
            content="grandparent content",
            original_span=(0, 150),
            hierarchy_level=0,
            parent_id=None,
    )
    return ChunkedDocument(
        filename="multi-layered document",
        chunks=[grant_parent] + create_kids(grant_parent, 2),
    )


def test_operations_level_all(multilayered_chunked_document):
    for chunk in multilayered_chunked_document.chunk_iterator(operation_level=None):
        assert 0 <= chunk.hierarchy_level <= 2


def test_operations_level_none(multilayered_chunked_document):
    count = sum(
        1
        for _ in multilayered_chunked_document.chunk_iterator(operation_level=3)
    )
    assert count == 0

def test_operations_level_two(multilayered_chunked_document):
    count = sum(
        1
        for _ in multilayered_chunked_document.chunk_iterator(operation_level=2)
    )
    assert count == 25

def test_operations_level_lowest(multilayered_chunked_document):
    count = 0
    for chunk in multilayered_chunked_document.chunk_iterator(operation_level=-1):
        count += 1
        assert chunk.hierarchy_level == 2
    assert count == 25