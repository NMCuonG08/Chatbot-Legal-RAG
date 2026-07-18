"""Unit tests for cost_routing: route->model selection + savings estimate."""
import pytest

from evaluation import cost_routing as cr


def test_general_chat_routes_to_small():
    rule = cr.select_model_for_route("general_chat")
    assert rule.model == cr.SMALL_MODEL
    assert rule.route == "general_chat"


def test_agent_tools_routes_to_small():
    rule = cr.select_model_for_route("agent_tools")
    assert rule.model == cr.SMALL_MODEL


def test_legal_rag_routes_to_big():
    rule = cr.select_model_for_route("legal_rag")
    assert rule.model == cr.BIG_MODEL


def test_hard_legal_rag_keeps_big():
    rule = cr.select_model_for_route("legal_rag", difficulty="hard")
    assert rule.model == cr.BIG_MODEL
    assert "hard" in rule.reason


def test_unknown_route_falls_back_to_general():
    rule = cr.select_model_for_route("nonsense_route")
    assert rule.route == "general_chat"
    assert rule.model == cr.SMALL_MODEL


def test_none_route_falls_back_to_general():
    rule = cr.select_model_for_route(None)
    assert rule.route == "general_chat"


def test_savings_positive_for_small_route():
    s = cr.estimate_route_savings("general_chat")
    assert s > 0.0


def test_savings_zero_for_legal_rag():
    s = cr.estimate_route_savings("legal_rag")
    assert s == 0.0


def test_cost_route_rule_frozen():
    rule = cr.select_model_for_route("general_chat")
    with pytest.raises(Exception):
        rule.model = "x"