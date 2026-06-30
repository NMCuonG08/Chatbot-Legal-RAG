# RAG Evaluation Report
- generated_at: 2026-06-30T03:41:05.714080Z
- mode: all
- n_samples: 2
- seed: 42
- top_k: 10
## Retrieval ablation

| config | n | hit@1 | hit@3 | hit@5 | hit@10 | mrr | ndcg@10 | lat_mean_ms | lat_p95_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vector | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 10311.0 | 16539.6 |
| hybrid | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 9649.7 | 11087.1 |
| hybrid_rerank | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 8615.5 | 9019.9 |
| multi_query | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 25120.4 | 25218.0 |
| full | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 24967.3 | 25011.5 |

## Generation quality (LLM-as-judge)

| metric | value |
| --- | --- |
| n_queries | 2 |
| faithfulness_mean | 0.000 |
| answer_relevance_mean | 0.500 |
| context_precision_mean | 0.000 |
| latency_ms_mean | 46762.7 |
| latency_ms_p95 | 46920.8 |

## End-to-end (chat graph)

| metric | value |
| --- | --- |
| n_queries | 2 |
| success_rate | 1.000 |
| latency_ms_mean | 68668.9 |
| latency_ms_p95 | 78469.1 |

## Operational Metrics (Tokens & Cost)

| Stage / Run Mode | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost (USD) |
| --- | --- | --- | --- | --- |
| Generation Stage | 1859 | 876 | 2735 | $0.00016 |
| E2E Chat Graph Stage | 2192 | 578 | 2770 | $0.00016 |

## Failure Analysis Breakdown

| Failure Mode | Percentage |
| --- | --- |
| Success | 0.0% |
| Execution Error | 0.0% |
| Routing Failure | 0.0% |
| Retrieval Failure | 100.0% |
| Hallucination Failure | 0.0% |
| Irrelevance Failure | 0.0% |

## Agentic & Routing Metrics

### Routing Decisions
| Routing Metric | Value |
| --- | --- |
| Routing Accuracy (Expected vs Actual) | 100.0% |
| Route Chosen: agent_tools | 1 |
| Route Chosen: legal_rag | 1 |

### ReAct Agent Tool Usage
| Agentic Tool Metric | Value |
| --- | --- |
| Total Tool Calls | 4 |
| Tool Calls Success Rate | 100.0% |
| Tool Used: web_search_tool | 2 |
| Tool Used: tavily_search_tool | 2 |
