from app.retrieval.fusion import reciprocal_rank_fusion


def test_item_ranked_first_in_both_lists_wins():
    keyword = [1, 2, 3]
    vector = [1, 3, 2]
    fused = reciprocal_rank_fusion([keyword, vector], top_k=3)
    assert fused[0][0] == 1


def test_item_present_in_both_lists_outranks_a_single_list_item_at_rank_zero():
    keyword = [1, 2, 3]
    vector = [4, 5, 1]  # item 1 also appears here, at rank 2
    fused = reciprocal_rank_fusion([keyword, vector], top_k=5)
    ids = [chunk_id for chunk_id, _ in fused]
    assert ids[0] == 1  # present in both lists beats being rank 0 in only one


def test_top_k_is_respected():
    keyword = [1, 2, 3, 4, 5]
    fused = reciprocal_rank_fusion([keyword], top_k=2)
    assert len(fused) == 2


def test_disjoint_lists_still_fuse_without_error():
    fused = reciprocal_rank_fusion([[1, 2], [3, 4]], top_k=10)
    assert {c for c, _ in fused} == {1, 2, 3, 4}


def test_empty_lists_return_empty():
    assert reciprocal_rank_fusion([[], []], top_k=5) == []
