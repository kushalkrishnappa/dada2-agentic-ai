import types

import pytest

from rag import retrieval as R


def test_filter_company_and_year_are_optional():
    f = R.Filter(search="TSMC dependence")
    assert f.ticker is None and f.fiscal_year is None
    g = R.Filter(ticker="NVDA", fiscal_year=2026, search="export controls")
    assert (g.ticker, g.fiscal_year) == ("NVDA", 2026)


def test_where_of_both_fields():
    f = R.Filter(ticker="NVDA", fiscal_year=2026, search="x")
    assert R.where_of(f) == {"ticker": {"$eq": "NVDA"}, "fiscal_year": {"$eq": 2026}}


def test_where_of_ticker_only():
    assert R.where_of(R.Filter(ticker="MU", search="x")) == {"ticker": {"$eq": "MU"}}


def test_where_of_nothing_is_none_not_empty_dict():
    assert R.where_of(R.Filter(search="x")) is None


def test_extract_filter_sends_the_question(fake_chat):
    want = R.Filter(ticker="AMD", fiscal_year=2025, search="supply")
    chat = fake_chat({R.Filter: want})
    assert R.extract_filter("What did AMD say about supply in fiscal 2025?") is want
    assert "What did AMD say about supply in fiscal 2025?" in chat.prompts[0]
    assert "exactly one" in chat.prompts[0]


def test_write_queries_caps_at_n(fake_chat):
    fake_chat({R.Queries: R.Queries(queries=["a", "b", "c", "d", "e"])})
    assert R.write_queries("x", n=4) == ["a", "b", "c", "d"]


def test_write_queries_falls_back_to_text(fake_chat):
    fake_chat({R.Queries: R.Queries(queries=[])})
    assert R.write_queries("original") == ["original"]


def test_rrf_matches_the_hand_calculation():
    # X: 1/61 + 1/63, Y: 1/62 + 1/61, W: 1/62, Z: 1/63
    assert R.rrf([["X", "Y", "Z"], ["Y", "W", "X"]]) == ["Y", "X", "W", "Z"]


def test_rrf_uses_positions_not_lists_order():
    assert R.rrf([["A"], ["B", "A"]], k=60) == ["A", "B"]


@pytest.fixture
def fake_dense(monkeypatch):
    calls = []

    def dense(query, k, where=None):
        calls.append((query, k, where))
        return [(f"{query}.txt", f"text for {query} {i}") for i in range(k)]
    monkeypatch.setattr(R, "dense", dense)
    return calls


@pytest.fixture
def nvda_filter(monkeypatch):
    f = R.Filter(ticker="NVDA", fiscal_year=2026, search="export controls")
    monkeypatch.setattr(R, "extract_filter", lambda q: f)
    return f


def test_with_filter_searches_f_search_with_where(fake_dense, nvda_filter):
    hits = R.with_filter("the question", 5)
    assert fake_dense == [("export controls", 5, R.where_of(nvda_filter))]
    assert len(hits) == 5 and all(isinstance(h, tuple) for h in hits)


def test_filter_rrf_one_filtered_search_per_rewrite(fake_dense, nvda_filter, monkeypatch):
    monkeypatch.setattr(R, "write_queries", lambda text, n=4: ["q1", "q2", "q3"])
    hits = R.filter_rrf("the question", 5)
    assert [c[0] for c in fake_dense] == ["q1", "q2", "q3"]
    assert all(c[2] == R.where_of(nvda_filter) for c in fake_dense)
    assert len(hits) == 5


def test_filter_rrf_rerank_pools_then_reranks_the_original_question(fake_dense, nvda_filter, monkeypatch):
    monkeypatch.setattr(R, "write_queries", lambda text, n=4: ["q1", "q2", "q3", "q4"])
    seen = {}

    def rerank(question, passages, top_n=5):
        seen.update(question=question, pool=len(passages), top_n=top_n)
        return passages[:top_n]
    monkeypatch.setattr(R, "rerank", rerank)
    hits = R.filter_rrf_rerank("the original question", 5)
    assert seen == {"question": "the original question", "pool": R.config.POOL, "top_n": 5}
    assert len(hits) == 5


def test_rerank_cohere_maps_result_indexes(monkeypatch):
    import cohere
    sent = {}

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def rerank(self, **kwargs):
            sent.update(kwargs)
            return types.SimpleNamespace(results=[types.SimpleNamespace(index=2),
                                                  types.SimpleNamespace(index=0)])
    monkeypatch.setattr(cohere, "ClientV2", Client)
    monkeypatch.setenv("COHERE_API_KEY", "test")
    passages = [("a.txt", "A"), ("b.txt", "B"), ("c.txt", "C")]
    assert R.rerank_cohere("q", passages, top_n=2) == [("c.txt", "C"), ("a.txt", "A")]
    assert sent["model"] == "rerank-v3.5" and sent["query"] == "q"
    assert sent["documents"] == ["A", "B", "C"] and sent["top_n"] == 2


def test_rerank_cohere_empty_pool_makes_no_call(monkeypatch):
    import cohere
    monkeypatch.setattr(cohere, "ClientV2", lambda *a, **k: pytest.fail("called"))
    assert R.rerank_cohere("q", [], top_n=5) == []


def test_rerank_cohere_backs_off_on_a_429(monkeypatch):
    import cohere
    from cohere.errors import TooManyRequestsError
    from rag import util
    calls, slept = [], []

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def rerank(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise TooManyRequestsError(body={})
            return types.SimpleNamespace(results=[types.SimpleNamespace(index=0)])
    monkeypatch.setattr(cohere, "ClientV2", Client)
    monkeypatch.setattr(util.time, "sleep", slept.append)
    monkeypatch.setenv("COHERE_API_KEY", "test")
    assert R.rerank_cohere("q", [("a.txt", "A")], top_n=1) == [("a.txt", "A")]
    assert len(calls) == 2 and len(slept) == 1
