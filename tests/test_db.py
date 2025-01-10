from genie_flow_invoker.invoker.docproc.similarity_search.db import VectorDB, ChunkVector


def test_db_empty():
    db = VectorDB([])

    vectors = db.get_vectors()
    assert isinstance(vectors, list)
    assert len(vectors) == 0


def test_db_all(multilayered_chunked_document):
    db = VectorDB(multilayered_chunked_document.chunks)

    vectors = db.get_vectors()
    assert isinstance(vectors, list)
    assert len(vectors) == len(multilayered_chunked_document.chunks)
    for i, vector in enumerate(vectors):
        assert isinstance(vector, ChunkVector)
        assert vector.chunk.chunk_id == multilayered_chunked_document.chunks[i].chunk_id


def test_db_top(multilayered_chunked_document):
    db = VectorDB(multilayered_chunked_document.chunks)

    vectors = db.get_vectors(operation_level=0)
    assert isinstance(vectors, list)
    assert len(vectors) == 1
    for i, vector in enumerate(vectors):
        assert isinstance(vector, ChunkVector)
        assert vector.chunk.chunk_id == multilayered_chunked_document.chunks[i].chunk_id


def test_db_middle(multilayered_chunked_document):
    db = VectorDB(multilayered_chunked_document.chunks)

    vectors = db.get_vectors(operation_level=1)
    assert isinstance(vectors, list)

    vector_ids = [vector.chunk.chunk_id for vector in vectors]
    assert len(vectors) == sum(
        1
        for chunk in multilayered_chunked_document.chunks
        if (
            chunk.hierarchy_level == 1
            and chunk.chunk_id in vector_ids
        )
    )


def test_db_bottom(multilayered_chunked_document):
    db = VectorDB(multilayered_chunked_document.chunks)

    vectors = db.get_vectors(operation_level=-1)
    assert isinstance(vectors, list)

    vector_ids = [vector.chunk.chunk_id for vector in vectors]
    assert len(vectors) == sum(
        1
        for chunk in multilayered_chunked_document.chunks
        if (
            chunk.hierarchy_level == 2
            and chunk.chunk_id in vector_ids
        )
    )


def test_db_middle_from_bottom(multilayered_chunked_document):
    db = VectorDB(multilayered_chunked_document.chunks)

    vectors = db.get_vectors(operation_level=-2)
    assert isinstance(vectors, list)

    vector_ids = [vector.chunk.chunk_id for vector in vectors]
    assert len(vectors) == sum(
        1
        for chunk in multilayered_chunked_document.chunks
        if (
            chunk.hierarchy_level == 1
            and chunk.chunk_id in vector_ids
        )
    )
