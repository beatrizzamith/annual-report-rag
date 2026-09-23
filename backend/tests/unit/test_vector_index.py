from app.retrieval.vector_index import VectorIndex


def test_remove_drops_the_given_chunks_and_keeps_the_rest():
    index = VectorIndex(dim=4)
    index.add(1, [1.0, 0.0, 0.0, 0.0])
    index.add(2, [0.0, 1.0, 0.0, 0.0])
    index.add(3, [0.0, 0.0, 1.0, 0.0])

    index.remove({2})

    assert index.size == 2
    results = index.search([1.0, 0.0, 0.0, 0.0], candidate_ids=None, top_k=10)
    assert {chunk_id for chunk_id, _ in results} == {1, 3}


def test_remove_with_unknown_ids_is_a_no_op():
    index = VectorIndex(dim=4)
    index.add(1, [1.0, 0.0, 0.0, 0.0])

    index.remove({999})

    assert index.size == 1


def test_remove_with_empty_set_is_a_no_op():
    index = VectorIndex(dim=4)
    index.add(1, [1.0, 0.0, 0.0, 0.0])

    index.remove(set())

    assert index.size == 1
