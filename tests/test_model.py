
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