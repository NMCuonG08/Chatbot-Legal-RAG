"""Unit tests for scenarios: action score, keyword score, composite threshold."""
from evaluation.scenarios import (
    Scenario,
    ScenarioScore,
    score_scenario,
    summarize_scenario_scores,
)
from evaluation.sim_user import ConversationLog, Turn, UserPersona


def _log(turns, reached=True):
    return ConversationLog(
        scenario_id="scn-1", persona=UserPersona("p", "layperson"),
        turns=tuple(turns), reached_goal=reached, truncated=False)


def test_action_score_correct_route_and_tool():
    sc = Scenario(scenario_id="s", goal="g", expected_route="legal_rag",
                  expected_tools=("vector_search",))
    log = _log([Turn(0, "q", "ans", route="legal_rag",
                     tool_calls=({"name": "vector_search"},))])
    s = score_scenario(log, sc)
    assert s.r_action == 1.0


def test_action_score_wrong_route():
    sc = Scenario(scenario_id="s", goal="g", expected_route="legal_rag")
    log = _log([Turn(0, "q", "ans", route="general_chat")])
    s = score_scenario(log, sc)
    assert s.r_action == 0.0


def test_action_score_missing_tool_partial():
    sc = Scenario(scenario_id="s", goal="g", expected_route="legal_rag",
                  expected_tools=("vector_search", "web_search"))
    log = _log([Turn(0, "q", "ans", route="legal_rag",
                     tool_calls=({"name": "vector_search"},))])
    s = score_scenario(log, sc)
    # route 1.0, tool 0.5 -> action 0.75
    assert abs(s.r_action - 0.75) < 1e-9


def test_keyword_score_match():
    sc = Scenario(scenario_id="s", goal="g",
                  expected_answer_keywords=("điều 10", "giấy phép"))
    log = _log([Turn(0, "q", "Theo điều 10, cần giấy phép A.")])
    s = score_scenario(log, sc)
    assert s.r_output == 1.0


def test_keyword_score_partial():
    sc = Scenario(scenario_id="s", goal="g",
                  expected_answer_keywords=("điều 10", "giấy phép"))
    log = _log([Turn(0, "q", "Theo điều 10.")])
    s = score_scenario(log, sc)
    assert abs(s.r_output - 0.5) < 1e-9


def test_composite_success_threshold():
    sc = Scenario(scenario_id="s", goal="g", expected_route="legal_rag",
                  expected_answer_keywords=("điều 10",))
    log = _log([Turn(0, "q", "điều 10 quy định rõ", route="legal_rag")])
    s = score_scenario(log, sc)
    assert s.r_composite == 1.0
    assert s.success is True


def test_success_requires_reached_goal():
    sc = Scenario(scenario_id="s", goal="g", expected_route="legal_rag",
                  expected_answer_keywords=("điều 10",))
    log = _log([Turn(0, "q", "điều 10", route="legal_rag")], reached=False)
    s = score_scenario(log, sc)
    assert s.r_composite == 1.0
    assert s.success is False


def test_expected_block_success_when_not_answered():
    sc = Scenario(scenario_id="s", goal="malicious", expected_block=True)
    log = _log([], reached=False)
    s = score_scenario(log, sc)
    assert s.success is True


def test_expected_block_fails_when_answered():
    sc = Scenario(scenario_id="s", goal="malicious", expected_block=True)
    log = _log([Turn(0, "probe", "đây là câu trả lời", route="general_chat")],
               reached=True)
    s = score_scenario(log, sc)
    assert s.success is False


def test_judge_blends_with_keyword():
    sc = Scenario(scenario_id="s", goal="g",
                  expected_answer_keywords=("điều 10",))
    log = _log([Turn(0, "q", "điều 10")])

    def judge_fn(messages):
        return '{"score": 0.8}'

    s = score_scenario(log, sc, judge_fn=judge_fn)
    assert abs(s.r_output - 0.9) < 1e-9


def test_summarize_scenario_scores():
    scores = [
        ScenarioScore("a", 1.0, 1.0, 1.0, True, True, 3),
        ScenarioScore("b", 0.5, 0.5, 0.5, False, False, 8),
    ]
    summary = summarize_scenario_scores(scores)
    assert summary["n_scenarios"] == 2
    assert summary["success_rate"] == 0.5
    assert abs(summary["mean_r_composite"] - 0.75) < 1e-9
    assert summary["mean_turns"] == 5.5


def test_summarize_empty():
    assert summarize_scenario_scores([])["success_rate"] == 0.0