import pytest

from rag import agent as A
from rag import retrieval

P = {"sufficient": 0.1, "absent": 0.1, "wrong_words": 0.1, "incomplete": 0.1}


def probs(**over):
    return {**P, **over}


# ------------------------------------------------------------- 3.1 route
def test_route_sure_uses_start_with(monkeypatch):
    monkeypatch.setattr(A.decision, "scope", lambda q: ("comparative", {"comparative": 0.9}))
    assert A.route({"question": "q"}) == {"scope": "comparative", "strategy": "hybrid+rrf",
                                          "decision_calls": 1}


def test_route_unsure_uses_default(monkeypatch):
    monkeypatch.setattr(A.decision, "scope", lambda q: ("specific", {"specific": 0.59}))
    assert A.route({"question": "q"})["strategy"] == A.DEFAULT


def test_route_at_exactly_sure_is_sure(monkeypatch):
    monkeypatch.setattr(A.decision, "scope", lambda q: ("specific", {"specific": A.SURE}))
    assert A.route({"question": "q"})["strategy"] == "filter"


def test_after_route():
    assert A.after_route({"scope": "figure"}) == "lookup"
    assert A.after_route({"scope": "specific"}) == "retrieve"
    assert A.after_route({"scope": "comparative"}) == "retrieve"


# ------------------------------------------------------------- 3.2 lookup
def test_lookup_names_the_figure_and_reads_the_file(fake_chat):
    chat = fake_chat({A.Fact: A.Fact(ticker="MU", concept="ResearchAndDevelopmentExpense")})
    out = A.lookup({"question": "What did Micron spend on R&D in fiscal 2025?"})
    assert out["verdict"] == "sufficient" and out["attempts"] == 1 and out["chat_calls"] == 1
    [(source, table)] = out["documents"]
    assert source == "MU_financials.json"
    assert table.startswith("MU ResearchAndDevelopmentExpense:")
    assert "What did Micron spend on R&D in fiscal 2025?" in chat.prompts[0]


# ------------------------------------------------------------- 3.3 diagnose
def test_diagnose_passes_verdict_through(monkeypatch):
    monkeypatch.setattr(A.decision, "diagnose", lambda q, d: ("sufficient", probs(sufficient=0.9)))
    out = A.diagnose({"question": "q", "documents": [("a", "b")], "attempts": 1})
    assert out == {"verdict": "sufficient", "probabilities": probs(sufficient=0.9),
                   "decision_calls": 1}


def test_diagnose_unsure_absent_becomes_wrong_words(monkeypatch):
    monkeypatch.setattr(A.decision, "diagnose", lambda q, d: ("absent", probs(absent=0.79)))
    assert A.diagnose({"question": "q", "documents": [], "attempts": 1})["verdict"] == "wrong_words"


def test_diagnose_sure_absent_stays_absent(monkeypatch):
    monkeypatch.setattr(A.decision, "diagnose", lambda q, d: ("absent", probs(absent=0.8)))
    assert A.diagnose({"question": "q", "documents": [], "attempts": 1})["verdict"] == "absent"


def test_diagnose_unsure_absent_with_no_attempts_left_stays_absent(monkeypatch):
    monkeypatch.setattr(A.decision, "diagnose", lambda q, d: ("absent", probs(absent=0.5)))
    state = {"question": "q", "documents": [], "attempts": A.MAX_ATTEMPTS}
    assert A.diagnose(state)["verdict"] == "absent"


# ------------------------------------------------------------- 3.3 decide
@pytest.mark.parametrize("state, nxt", [
    ({"verdict": "sufficient", "attempts": 1, "strategy": "filter"}, "answer"),
    ({"verdict": "absent", "attempts": 1, "strategy": "filter"}, "answer"),
    ({"verdict": "sufficient", "attempts": 3, "strategy": "filter"}, "answer"),
    ({"verdict": "incomplete", "attempts": 3, "strategy": "filter"}, "answer"),
    ({"verdict": "wrong_words", "attempts": 3, "strategy": "filter"}, "answer"),
    ({"verdict": "incomplete", "attempts": 1, "strategy": "hybrid+rrf"}, "plan"),
    ({"verdict": "wrong_words", "attempts": 1, "strategy": "filter"}, "stronger"),
    ({"verdict": "wrong_words", "attempts": 2, "strategy": "filter+rrf+rerank"}, "answer"),
])
def test_decide(state, nxt):
    assert A.decide(state) == nxt


# ------------------------------------------------------------- 3.4 nodes
def test_stronger():
    assert A.stronger({"strategy": "filter"}) == {"strategy": "filter+rrf"}
    assert A.stronger({"strategy": "hybrid+rrf"}) == {"strategy": "filter+rrf+rerank"}


def test_plan_says_what_is_covered(fake_chat):
    chat = fake_chat({A.Plan: A.Plan(steps=["NVIDIA TSMC dependence"])})
    out = A.plan({"question": "q", "documents": [("B.txt", "x"), ("A.txt", "y"), ("B.txt", "z")]})
    assert out == {"steps": ["NVIDIA TSMC dependence"], "chat_calls": 1}
    assert "Passages found so far cover: A.txt, B.txt" in chat.prompts[0]


def test_plan_with_nothing_found(fake_chat):
    chat = fake_chat({A.Plan: A.Plan(steps=[])})
    A.plan({"question": "q", "documents": []})
    assert "Passages found so far cover: nothing" in chat.prompts[0]


def test_execute_keeps_only_new_hits(monkeypatch):
    hits = {"s1": [("A.txt", "a"), ("N.txt", "n")], "s2": [("N.txt", "n"), ("M.txt", "m")]}
    monkeypatch.setattr(retrieval, "with_filter", lambda step, k: hits[step])
    out = A.execute({"steps": ["s1", "s2"], "documents": [("A.txt", "a")], "attempts": 1})
    assert out == {"documents": [("N.txt", "n"), ("M.txt", "m")], "attempts": 2, "chat_calls": 2}


# ------------------------------------------------------------- the graph
EDGES = {("__start__", "route"), ("route", "lookup"), ("route", "retrieve"),
         ("lookup", "answer"), ("retrieve", "diagnose"), ("diagnose", "answer"),
         ("diagnose", "stronger"), ("diagnose", "plan"), ("stronger", "retrieve"),
         ("plan", "execute"), ("execute", "diagnose"), ("answer", "__end__")}


def test_graph_has_exactly_the_diagram_edges():
    g = A.build_graph().get_graph()
    assert {(e.source, e.target) for e in g.edges} == EDGES


def test_graph_absent_path(monkeypatch, fake_chat):
    fake_chat()
    monkeypatch.setattr(A.decision, "scope", lambda q: ("specific", {"specific": 0.9}))
    monkeypatch.setattr(A.decision, "diagnose", lambda q, d: ("absent", probs(absent=0.95)))
    monkeypatch.setitem(retrieval.STRATEGIES, "filter", lambda q, k: [("A.txt", "a")])
    s = A.run("What is AMD's dress code?")
    assert s["answer"] == "The filings in this corpus do not cover that."
    assert (s["chat_calls"], s["decision_calls"], s["attempts"]) == (1, 2, 1)


def test_graph_incomplete_then_sufficient(monkeypatch, fake_chat):
    fake_chat({A.Plan: A.Plan(steps=["NVIDIA TSMC"])}, text="both depend on TSMC")
    verdicts = [("incomplete", probs(incomplete=0.9)), ("sufficient", probs(sufficient=0.9))]
    monkeypatch.setattr(A.decision, "scope", lambda q: ("comparative", {"comparative": 0.9}))
    monkeypatch.setattr(A.decision, "diagnose", lambda q, d: verdicts.pop(0))
    monkeypatch.setitem(retrieval.STRATEGIES, "hybrid+rrf", lambda q, k: [("AMD.txt", "a")])
    monkeypatch.setattr(retrieval, "with_filter",
                        lambda step, k: [("NVDA.txt", "n"), ("AMD.txt", "a")])
    s = A.run("How do AMD and NVIDIA describe their dependence on TSMC?")
    assert s["documents"] == [("AMD.txt", "a"), ("NVDA.txt", "n")]
    assert s["answer"] == "both depend on TSMC"
    # hybrid 1 + plan 1 + execute 1 + answer 1; route 1 + diagnose 2
    assert (s["chat_calls"], s["decision_calls"], s["attempts"]) == (4, 3, 2)


def test_graph_figure_path_skips_retrieval(monkeypatch, fake_chat):
    fake_chat({A.Fact: A.Fact(ticker="MU", concept="ResearchAndDevelopmentExpense")})
    monkeypatch.setattr(A.decision, "scope", lambda q: ("figure", {"figure": 0.9}))
    monkeypatch.setattr(A.decision, "diagnose", lambda q, d: pytest.fail("no diagnosis"))
    s = A.run("What did Micron spend on R&D in fiscal 2025?")
    assert s["documents"][0][0] == "MU_financials.json"
    assert (s["chat_calls"], s["decision_calls"]) == (2, 1)


def test_graph_wrong_words_climbs_to_the_top_then_answers(monkeypatch, fake_chat):
    fake_chat()
    monkeypatch.setattr(A.decision, "scope", lambda q: ("specific", {"specific": 0.9}))
    monkeypatch.setattr(A.decision, "diagnose", lambda q, d: ("wrong_words", probs(wrong_words=0.9)))
    for name in ("filter", "filter+rrf", "filter+rrf+rerank"):
        monkeypatch.setitem(retrieval.STRATEGIES, name, lambda q, k, name=name: [(f"{name}.txt", name)])
    s = A.run("q")
    assert s["strategy"] == "filter+rrf+rerank" and len(s["documents"]) == 3
    # filter 1 + filter+rrf 2 + rerank 2 + answer 1; route 1 + diagnose 3
    assert (s["chat_calls"], s["decision_calls"], s["attempts"]) == (6, 4, 3)


def test_graph_attempt_cap_stops_a_persistent_incomplete(monkeypatch, fake_chat):
    fake_chat({A.Plan: A.Plan(steps=["NVIDIA TSMC"])})
    monkeypatch.setattr(A.decision, "scope", lambda q: ("comparative", {"comparative": 0.9}))
    monkeypatch.setattr(A.decision, "diagnose", lambda q, d: ("incomplete", probs(incomplete=0.9)))
    monkeypatch.setitem(retrieval.STRATEGIES, "hybrid+rrf", lambda q, k: [("AMD.txt", "a")])
    monkeypatch.setattr(retrieval, "with_filter", lambda step, k: [("NVDA.txt", "n")])
    s = A.run("q")
    assert s["documents"] == [("AMD.txt", "a"), ("NVDA.txt", "n")]
    # hybrid 1 + (plan 1 + execute 1) x2 + answer 1; route 1 + diagnose 3
    assert (s["chat_calls"], s["decision_calls"], s["attempts"]) == (6, 4, 3)


# ------------------------------------------------------------- the agent's tool call

def test_search_filings_question_keeps_company_and_year():
    # The outer model writes the tool argument. Without a description it
    # sometimes shortens the question and drops the fiscal year, so the
    # filter can only pin the company.
    from langchain_core.utils.function_calling import convert_to_openai_tool
    spec = convert_to_openai_tool(A.search_filings)["function"]["parameters"]
    described = spec["properties"]["question"].get("description", "").lower()
    assert "company" in described and "fiscal year" in described
    assert "earlier turns" in described


def test_system_prompt_says_the_search_filters_on_company_and_year():
    prompt = A.SYSTEM_PROMPT.lower()
    assert "search_filings" in prompt and "fiscal year" in prompt and "filters" in prompt
