from app.core.rerank import lexical_overlap_score


def test_lexical_overlap_prefers_matching_chunk():
    query = "project deadline and budget"
    matching = "The project deadline is March and the budget is $50k."
    unrelated = "Weather forecast shows rain tomorrow."
    assert lexical_overlap_score(query, matching) > lexical_overlap_score(query, unrelated)


def test_lexical_overlap_empty_query_returns_zero():
    assert lexical_overlap_score("", "some text") == 0.0
