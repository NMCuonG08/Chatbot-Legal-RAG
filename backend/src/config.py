DEFAULT_COLLECTION_NAME = "llm"

# ---- Self-Corrective RAG (CRAG) tuning constants ----
# Max retrieval-rewrite loop iterations before forcing a web_search fallback.
REFLECTION_MAX = 2
# rerank relevance_score >= this -> doc graded "relevant" without LLM judge.
DOC_GRADE_THRESHOLD = 0.35
# If fewer than this many docs are relevant, consider retrieval "all irrelevant".
ALL_IRRELEVANT_THRESHOLD = 1