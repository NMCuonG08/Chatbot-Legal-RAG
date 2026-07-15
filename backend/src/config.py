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