DEFAULT_COLLECTION_NAME = "llm"

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