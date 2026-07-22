import os as _os_loop  # loop-control env reads before the shared `import os as _os` block below

DEFAULT_COLLECTION_NAME = "llm"

# ---- LangGraph loop control (senior: bound the loop, never hang) ----
# Hard recursion cap on the graph. LangGraph default is 25; we raise slightly
# because supervisor handoffs + ReAct + verify loops can legitimately stack a few
# layers. Anything exceeding this is a runaway loop -> GraphRecursionError
# (caught + degraded, NOT retried). Set via env so ops can tune per-env.
GRAPH_RECURSION_LIMIT = int(_os_loop.environ.get("GRAPH_RECURSION_LIMIT", "32"))
# Wall-clock deadline for one graph.invoke (seconds). Per-LLM request_timeout=60
# already bounds a single model call, but a chain of nodes + a stuck tool could
# still hang a Celery worker indefinitely. On timeout we raise GraphRunTimeout
# (non-retryable) and degrade gracefully. The hung invoke thread cannot be
# killed in Python; it is abandoned and left to fail on its own — the worker
# unblocks and returns a user-facing error.
# Default lowered 120 -> 60s: a healthy legal chat rarely exceeds 30-40s; the
# ReAct agent path has its own tighter AGENT_RUN_TIMEOUT_S cap below, so 60s is
# a safety net for the whole graph (supervisor + verify + meta), not the agent.
GRAPH_RUN_TIMEOUT_S = float(_os_loop.environ.get("GRAPH_RUN_TIMEOUT_S", "180"))

# ---- Ollama Model Tiers (Fast vs Reasoning) ----
OLLAMA_LLM_MODEL = _os_loop.environ.get("OLLAMA_LLM_MODEL", "gemma4:31b")
OLLAMA_FAST_LLM_MODEL = _os_loop.environ.get("OLLAMA_FAST_LLM_MODEL", "qwen2.5:7b")

# ---- ReAct agent latency guards (senior: bound the agent, kill empty-retry thrash) ----
# Hard wall-clock cap on one ai_agent_handle run (seconds). The ReAct loop is the
# dominant latency source: each round trip = 1 LLM call + reasoning tokens, and an
# agent retrying an empty-result search tool N times compounds to 60-90s. This cap
# wraps asyncio.run(agent) in asyncio.wait_for; on timeout the agent bails to a
# graceful fallback instead of lũy kế further. Tighter than GRAPH_RUN_TIMEOUT_S.
AGENT_RUN_TIMEOUT_S = float(_os_loop.environ.get("AGENT_RUN_TIMEOUT_S", "120"))
# ReAct max tool-call round trips. 10 (llama-index default) invites thrashing on
# empty results — legal queries need <=2 tool calls in practice. Lowered to 4:
# enough for lookup -> verify_citation -> (one reformulation), not enough to spin.
AGENT_MAX_ITERATIONS = int(_os_loop.environ.get("AGENT_MAX_ITERATIONS", "4"))
# Max consecutive empty-tool results before the @track_tool_call guard forces the
# agent to STOP retrying and synthesize a "no info found" answer. 2 = after the
# 2nd empty, the 3rd identical call is blocked with a reformulate-or-stop sentinel.
# Stops the "call -> empty -> think -> call same args -> empty -> think" lũy kế.
AGENT_MAX_EMPTY_STREAK = int(_os_loop.environ.get("AGENT_MAX_EMPTY_STREAK", "2"))

# ---- Pure-compute tool sandbox (defense-in-depth, OPT-IN) ----
# When true, the @sandboxable pure-compute tools (contract_penalty, pit,
# legal_age, ...) execute in a throwaway subprocess with a scrubbed env + hard
# timeout via sandbox.run_in_sandbox, instead of in-process. Audit 3.2: default
# ON so a bug in a complex calc (e.g. runaway inheritance loop) cannot wedge the
# agent/worker process. Input validation still runs first (@_validated is stacked
# outermost), so isolation does not bypass the Flaw 4 guard. Set SANDBOX_ENABLED=0
# to disable. Reads at call time so ops can flip it without a redeploy (and tests
# can monkeypatch / the conftest fixture forces it off for the unit suite).
SANDBOX_ENABLED = _os_loop.environ.get("SANDBOX_ENABLED", "1").lower() in ("1", "true", "yes", "on")

# ---- Self-Corrective RAG (CRAG) tuning constants ----
# Max retrieval-rewrite loop iterations before forcing a web_search fallback.
REFLECTION_MAX = 2
# rerank relevance_score >= this -> doc graded "relevant" without LLM judge.
DOC_GRADE_THRESHOLD = 0.35
# If fewer than this many docs are relevant, consider retrieval "all irrelevant".
ALL_IRRELEVANT_THRESHOLD = 1

# ---- PEV verify_answer (final-answer groundedness gate) ----
# Max verify->rewrite_query recovery loops before degrading to metacognitive.
VERIFY_MAX_RETRIES = 2
# evaluate_faithfulness score >= this -> verdict "supported" (pass the gate).
VERIFY_ANSWER_THRESHOLD = 0.7
# score >= this (but < supported) -> verdict "partial"; below -> "unsupported".
VERIFY_PARTIAL_THRESHOLD = 0.35

# ---- Metacognitive escalation (Phase 2) ----
# On a MEDIUM-stakes question, escalate to "consult a lawyer" only when the
# verify_answer confidence (groundedness score) drops below this threshold.
# HIGH-stakes questions always escalate regardless of confidence.
ESCALATION_CONFIDENCE_THRESHOLD = 0.6

# ---- RLHF rerank up-weight (Phase 4) ----
# Additive boost applied to a chunk's relevance_score when its doc_id backs a
# 👍-marked answer for the current user. Small so a good-answer source wins
# ties without overriding a clearly-more-relevant chunk.
RLHF_RERANK_BOOST = 0.05

# ---- Task-level failure/retry (lifecycle resilience) ----
# Celery autoretry knobs for idempotent tasks whose only realistic failure mode
# is transient infra (Redis/DB/Qdrant/HTTP down). Side-effectful tasks with
# non-idempotent writes (llm_handle_message) do NOT autoretry — they degrade
# gracefully instead — because a retry would duplicate the user message + reply.
TASK_MAX_RETRIES = 3
# Exponential backoff: Celery sleeps 1, 2, 4 ... seconds (capped) between
# attempts when retry_backoff=True. Jitter spreads a thundering herd of
# retries after a shared outage.
TASK_RETRY_BACKOFF = True
TASK_RETRY_BACKOFF_MAX = 60
TASK_RETRY_JITTER = True

# ---- Evaluation harness (Phase P1) ----
# Judge is pinned + separate from the agent model so a judge swap is auditable
# and regression diffs can attribute score changes to the judge vs. the agent.
# Groq + Ollama only (no OpenAI/Anthropic key in this project).
import os as _os
JUDGE_PROVIDER = _os.environ.get("JUDGE_PROVIDER", "groq").lower()
JUDGE_MODEL = _os.environ.get("JUDGE_MODEL", "llama-3.1-8b-instant")
JUDGE_TEMPERATURE = float(_os.environ.get("JUDGE_TEMPERATURE", "0.0"))
EVAL_MAX_WORKERS = int(_os.environ.get("EVAL_MAX_WORKERS", "8"))
EVAL_JUDGE_CONCURRENCY = int(_os.environ.get("EVAL_JUDGE_CONCURRENCY", "4"))
REGRESSION_ALPHA = float(_os.environ.get("REGRESSION_ALPHA", "0.05"))

# ---- PII output guardrail (Phase P4) ----
# Eval-time PII detection on agent outputs. Default OFF so production behavior
# is unchanged; set GUARDRAILS_PII_OUTPUT_ENABLED=true to redact/block PII in
# run_chat_graph output (post-verify). Homegrown regex — no presidio dependency.
GUARDRAILS_PII_OUTPUT_ENABLED = (
    _os.environ.get("GUARDRAILS_PII_OUTPUT_ENABLED", "false").lower() == "true"
)

# ---- OpenTelemetry bridge (Phase P6) ----
# Default OFF. When true + OTEL_EXPORTER_OTLP_ENDPOINT set, trace.py mirrors
# emit_* calls as OTel spans (fire-and-forget; MySQL+Redis trace unchanged).
OTEL_BRIDGE_ENABLED = (
    _os.environ.get("OTEL_BRIDGE_ENABLED", "false").lower() == "true"
)

# ---- Cost-aware routing + canary/shadow (Phase P6) ----
# When true, run_chat_graph picks the agent model per route (legal_rag -> big,
# others -> small) via cost_routing.select_model_for_route + LLM_MODEL_CONTEXTVAR.
COST_ROUTING_ENABLED = (
    _os.environ.get("COST_ROUTING_ENABLED", "false").lower() == "true"
)
# Shadow mode: run candidate variant alongside primary, persist both, return
# primary to user. Default OFF (doubles Groq cost — nightly/opt-in only).
SHADOW_MODE_ENABLED = (
    _os.environ.get("SHADOW_MODE_ENABLED", "false").lower() == "true"
)