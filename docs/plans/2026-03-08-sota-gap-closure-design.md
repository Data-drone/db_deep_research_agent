# Deep Research Loop — SOTA Gap Closure Design

**Date:** 2026-03-08
**Author:** Andy (AI) + Brian Law
**Branch:** feat/implementation
**Status:** Approved

## Overview

Close 10 identified gaps between our deep research loop and SOTA patterns (OpenAI Deep Research, STORM, Perplexity). Work is staged across 5 phases, each validated with MLflow Evaluate against a consistent eval dataset.

## Architecture: Dual-Model Split

Two LLM roles with different cost/quality profiles:

| Role | Model | Endpoint | Used By |
|------|-------|----------|---------|
| Worker | Claude Sonnet 4.6 | `databricks-claude-sonnet-4-6` | clarifier, planner, query_adapter, researcher, normalizer, compressor, synthesizer |
| Critic | GPT-5.4 | `databricks-gpt-5-4` | evaluator, verifier, evidence_scorer, perspective_generator |

The worker handles high-volume generation and tool orchestration. The critic handles judgment tasks where accuracy matters more than speed. This mirrors SOTA dual-model patterns where a stronger model evaluates a faster model's work.

## Current Graph

```
clarifier → planner → authorizer → researcher → normalizer → evaluator
                ↑                                                 |
                └──────────── (continue) ─────────────────────────┘
                                                                  |
                                                           (stop) ↓
                                                      compressor → synthesizer → verifier → END
```

## Target Graph (After All Phases)

```
clarifier → planner → authorizer → query_adapter → researcher (parallel) → normalizer → scorer → evaluator
                ↑                                                                                    |
                └──────────────────────── (continue) ────────────────────────────────────────────────┘
                                                                                                     |
                                                                                              (stop) ↓
                                                                     compressor → synthesizer → verifier
                                                                          ↑                        |
                                                                          └── (unsupported) ───────┘
                                                                                                   |
                                                                                        (all ok)   ↓
                                                                                                  END
```

Key changes: parallel researcher, query_adapter node, scorer node, verifier feedback loop, GPT-5.4 on evaluator/verifier/scorer.

## Eval Framework

### Eval Dataset

10-15 sample queries spanning all 4 MCP tools:

- Vector Search queries (ANZ annual report): factual recall, multi-hop
- Genie Aviation queries: data aggregation, trend analysis
- Genie Sales Pipeline queries: pipeline metrics, forecasting
- Knowledge Assistant queries: conceptual questions, cross-domain
- Multi-tool queries: questions requiring 2+ tools to answer fully

Saved as a Databricks table or JSON artifact in the MLflow experiment.

### Metrics

| Metric | Type | Judge |
|--------|------|-------|
| `answer_relevance` | Does the answer address the query? | GPT-5.4 |
| `citation_accuracy` | Are claims supported by cited evidence? | GPT-5.4 |
| `completeness` | Were all facets of the query covered? | GPT-5.4 |
| `tool_efficiency` | Useful evidence / total tool calls | Computed |
| `latency_p50` | Median end-to-end time | Computed |
| `latency_p95` | 95th percentile end-to-end time | Computed |
| `source_diversity` | Unique tools contributing evidence | Computed |

### Eval Harness

`eval_harness.py` script that:
1. Loads eval dataset
2. Runs each query through the graph
3. Collects outputs + intermediate state (evidence, tool calls, scores)
4. Calls `mlflow.evaluate()` with GPT-5.4 as judge
5. Logs results to MLflow experiment with phase tag

Each phase produces a tagged MLflow run for side-by-side comparison.

---

## Phase 0: Evaluation Baseline + Dual-Model Config

**Gaps addressed:** None (infrastructure only)
**Goal:** Establish measurable baseline before any changes.

### 0.1 Dual-Model Configuration

Add to `config.py`:
```python
@dataclass(frozen=True)
class AppConfig:
    ...
    llm_endpoint: str          # Worker: databricks-claude-sonnet-4-6
    critic_llm_endpoint: str   # Critic: databricks-gpt-5-4
```

Add to `main.py`:
```python
def _create_critic_model(config):
    from databricks_langchain import ChatDatabricks
    return ChatDatabricks(endpoint=config.critic_llm_endpoint)
```

Update `build_research_graph()` to accept both `model` (worker) and `critic_model` (critic). Nodes that use the critic receive it via `partial()`.

Environment variables:
- `LLM_ENDPOINT_NAME=databricks-claude-sonnet-4-6`
- `CRITIC_LLM_ENDPOINT=databricks-gpt-5-4`

### 0.2 Eval Dataset

Create `eval/eval_dataset.json` with 10-15 queries:
```json
[
  {
    "query": "What was ANZ's total revenue in 2024 and how did it compare to 2023?",
    "expected_tools": ["vector_search_anz"],
    "difficulty": "simple",
    "expected_facets": ["2024 revenue figure", "2023 comparison", "growth rate"]
  },
  ...
]
```

### 0.3 Eval Harness

Create `eval/eval_harness.py`:
- Runs queries through graph
- Uses `mlflow.evaluate()` with custom GPT-5.4 judge metrics
- Tags run as `phase=baseline`

### 0.4 Switch Worker Model

Update `app.yaml` env var from `databricks-meta-llama-3-1-70b-instruct` to `databricks-claude-sonnet-4-6`.

**Eval checkpoint:** Run baseline. All subsequent phases compare against this.

---

## Phase 1: Parallel Execution + Query Reformulation

**Gaps addressed:** #1 (parallel execution), #2 (query reformulation)
**Expected improvement:** Completeness +15-20%, tool efficiency +25%, latency -40%

### 1.1 Parallel Sub-Question Execution

Refactor `researcher_node` to fan out sub-questions concurrently:

```python
async def researcher_node(state, *, model, mcp_manager):
    pending = [sq for sq in plan if sq.status != "answered"]

    async def _research_one(sq):
        # Execute all assigned tools for this sub-question
        for tool_name in sq.assigned_tools:
            result = await mcp_manager.call_tool(tool_name, "query", ...)
            ...
        return evidence, tool_calls

    results = await asyncio.gather(
        *[_research_one(sq) for sq in pending],
        return_exceptions=True,
    )
    # Merge results, handle exceptions per-sub-question
```

Design decisions:
- Sub-questions run in parallel; tools within a sub-question run sequentially
- Budget check happens before launch (pre-allocate tool call slots)
- One sub-question failing does not kill the others
- `return_exceptions=True` for error isolation

### 1.2 Query Adapter Node

New node `query_adapter` between `authorizer` and `researcher`:

```python
async def query_adapter_node(state, *, model):
    """Reformulate each sub-question for its assigned tools."""
    plan = state.get("research_plan", [])
    for sq in plan:
        for tool_name in sq.assigned_tools:
            adapted = await _reformulate(model, sq.question, tool_name)
            sq.adapted_queries[tool_name] = adapted
    return {"research_plan": plan}
```

Reformulation strategy per tool type:
- **Genie (SQL):** "Rephrase as a natural-language data question suitable for SQL analysis"
- **Vector Search:** "Extract key terms and rephrase as a semantic search query"
- **Knowledge Assistant:** "Rephrase as a focused knowledge retrieval question"

Add `adapted_queries: dict[str, str]` field to `SubQuestion` model.

Researcher uses `sq.adapted_queries.get(tool_name, sq.question)` as the query.

### Graph Change

```
clarifier → planner → authorizer → query_adapter → researcher → normalizer → evaluator
```

**Eval checkpoint:** Run eval harness, tag as `phase=1`. Compare latency + completeness vs baseline.

---

## Phase 2: Smart Evaluation + Evidence Scoring

**Gaps addressed:** #3 (evaluator depth), #5 (sub-question status), #7 (evidence relevance)
**Expected improvement:** Answer relevance +20%, citation accuracy +15%

### 2.1 Evidence Relevance Scorer

New node `scorer` between `normalizer` and `evaluator`:

```python
async def scorer_node(state, *, critic_model):
    """Rate each evidence item's relevance to its sub-question using GPT-5.4."""
    for sq in plan:
        sq_evidence = [e for e in evidence if e.evidence_id in sq.evidence_ids]
        for ev in sq_evidence:
            score = await _score_relevance(critic_model, sq.question, ev.snippet)
            ev.confidence = score  # Replace hardcoded 0.8
    # Filter low-relevance evidence
    evidence = [e for e in evidence if e.confidence >= 0.3]
    return {"evidence": evidence}
```

This replaces the hardcoded 0.8 confidence (Gap #7).

### 2.2 Evaluator Uses Full Evidence + GPT-5.4

Switch evaluator to use `critic_model` (GPT-5.4):

```python
async def evaluator_node(state, *, critic_model):
    # Pass full evidence, not 200-char truncated
    evidence_text = "\n".join(
        f"- [{e.source_id}] (relevance={e.confidence:.2f}) {e.snippet}"
        for e in evidence
    )
```

Evaluator receives per-sub-question evidence bundles with relevance scores.

### 2.3 LLM-Assessed Sub-Question Status

Instead of marking "answered" when any evidence_id exists, GPT-5.4 evaluator assesses each sub-question:

```json
{
  "sub_question_verdicts": [
    {"id": "sq-abc", "verdict": "answered", "reason": "Revenue figures confirmed"},
    {"id": "sq-def", "verdict": "partially_answered", "reason": "Missing 2023 comparison"},
    {"id": "sq-ghi", "verdict": "unanswered", "reason": "No relevant evidence found"}
  ]
}
```

Add `partially_answered` to `SubQuestion.status` Literal. Replanner focuses on `partially_answered` and `unanswered` sub-questions.

### Graph Change

```
... → normalizer → scorer → evaluator → ...
```

Scorer uses critic_model. Evaluator uses critic_model.

**Eval checkpoint:** Run eval harness, tag as `phase=2`. Expect answer_relevance and citation_accuracy improvements.

---

## Phase 3: Verifier Feedback + Multi-Perspective + Source Diversity

**Gaps addressed:** #4 (perspectives), #8 (verifier feedback), #10 (source diversity)
**Expected improvement:** Completeness +15%, source diversity +30%

### 3.1 Verifier Feedback Loop

Verifier uses `critic_model` (GPT-5.4). Add conditional edge:

```python
def _should_revise(state):
    vr = state.get("verification_result")
    attempts = state.get("verification_attempts", 0)
    if vr and not vr.all_claims_supported and attempts < 2:
        return "planner"  # Loop back to fill gaps
    return END

graph.add_conditional_edges("verifier", _should_revise, {
    "planner": "planner",
    END: END,
})
```

Add `verification_attempts: int` to `ResearchState`. Planner receives `unsupported_claims` as new gaps.

### 3.2 Multi-Perspective Research (STORM-style)

Add perspective generation to planner using GPT-5.4:

```python
async def planner_node(state, *, model, critic_model):
    if iteration == 0:
        # GPT-5.4 generates 2-3 relevant perspectives
        perspectives = await _generate_perspectives(critic_model, query)
        # Worker model generates sub-questions from each perspective
        for perspective in perspectives:
            sub_qs = await _plan_from_perspective(model, query, perspective, tools)
            ...
```

Perspective examples:
- Financial query → "financial analyst", "risk manager", "retail investor"
- Technical query → "systems engineer", "product manager", "end user"
- General query → "domain expert", "skeptic", "synthesizer"

Perspectives stored in state: `perspectives: list[str]`. Each sub-question tagged with its perspective for traceability.

### 3.3 Source Diversity Tracking

Add diversity awareness to the evaluator:

```python
# In evaluator prompt
"For each sub-question, check if evidence comes from multiple tools.
 Flag sub-questions where all evidence comes from a single source."

# In evaluator output
"source_diversity_score": 0.7,
"single_source_questions": ["sq-abc relies only on vector_search"]
```

Replanner prioritizes corroboration from different tools for single-source questions.

### Graph Change

```
... → synthesizer → verifier ──→ END
                        |
                        └──→ planner (if unsupported claims, max 2 attempts)
```

**Eval checkpoint:** Run eval harness, tag as `phase=3`. Expect completeness and source diversity improvements.

---

## Phase 4: Cleanup — Clarifier + Authorizer

**Gaps addressed:** #6 (clarifier), #9 (authorizer)
**Expected improvement:** Minor quality-of-life; cleaner architecture

### 4.1 Clarifier Always Refines

Remove `clarification_needed` flag. Clarifier always produces the best rephrased query:

```python
async def clarifier_node(state, *, model):
    """Refine and focus the user query. Always produces a clarified version."""
    # Prompt: "Refine this query for research. Narrow scope if vague.
    #          Add implicit constraints (time range, geography, etc)."
    response = await model.ainvoke(...)
    return {"clarified_query": result.get("clarified_query")}
```

Remove `clarification_needed` from `ResearchState`.

### 4.2 Authorizer Implementation

```python
async def authorizer_node(state, *, model):
    """Validate tool access based on risk tiers and rate limits."""
    plan = state.get("research_plan", [])
    tool_call_log = state.get("tool_call_log", [])

    for sq in plan:
        approved_tools = []
        for tool in sq.assigned_tools:
            tier = _get_risk_tier(tool)
            if tier == "safe":
                approved_tools.append(tool)
            elif tier == "restricted":
                logger.info(f"Restricted tool access: {tool}")
                approved_tools.append(tool)
            elif tier == "privileged":
                # Check error rate — downgrade if failing > 50%
                if _tool_error_rate(tool, tool_call_log) > 0.5:
                    logger.warning(f"Downgrading {tool} — high error rate")
                    continue
                approved_tools.append(tool)
        sq.assigned_tools = approved_tools

    return {"research_plan": plan}
```

### Final Eval

Run full eval suite. Generate MLflow comparison:
- Phase 0 (baseline) → Phase 1 → Phase 2 → Phase 3 → Phase 4
- Side-by-side metrics across all phases
- Regression check: no metric should decrease by more than 5%

---

## Config Changes Summary

### app.yaml

```yaml
env:
  - name: LLM_ENDPOINT_NAME
    value: "databricks-claude-sonnet-4-6"
  - name: CRITIC_LLM_ENDPOINT
    value: "databricks-gpt-5-4"
```

### New Files

| File | Purpose |
|------|---------|
| `eval/eval_dataset.json` | Sample queries with expected facets |
| `eval/eval_harness.py` | MLflow Evaluate runner |
| `src/deep_research/nodes/query_adapter.py` | Per-tool query reformulation |
| `src/deep_research/nodes/scorer.py` | Evidence relevance scoring |

### Modified Files

| File | Changes |
|------|---------|
| `config.py` | Add `critic_llm_endpoint` to AppConfig |
| `main.py` | Add `_create_critic_model()`, pass both models to graph |
| `graph.py` | Add query_adapter, scorer nodes; verifier feedback loop |
| `models.py` | Add `adapted_queries` to SubQuestion, `partially_answered` status, `verification_attempts` + `perspectives` to state |
| `state.py` | Add `verification_attempts`, `perspectives`, `source_diversity_score` |
| `nodes/researcher.py` | Parallel execution with asyncio.gather |
| `nodes/planner.py` | Perspective generation, unsupported claims handling |
| `nodes/evaluator.py` | Full evidence, GPT-5.4, per-sub-question verdicts, diversity |
| `nodes/verifier.py` | Switch to critic_model |
| `nodes/normalizer.py` | No changes |
| `nodes/compressor.py` | No changes |
| `nodes/synthesizer.py` | No changes |
| `nodes/clarifier.py` | Remove clarification_needed, always refine |
| `nodes/authorizer.py` | Implement risk tier + error rate checks |
| `prompts.py` | Updated prompts for all changed nodes |

---

## Risk Mitigation

- **Latency:** Parallel execution (Phase 1) should offset added GPT-5.4 calls (Phase 2-3). Monitor p95.
- **Cost:** GPT-5.4 used only for judgment (evaluator, verifier, scorer, perspectives) — not high-volume tool calls.
- **Regression:** Each phase has an MLflow eval checkpoint. If a phase degrades metrics, it can be reverted independently.
- **Infinite loops:** Verifier feedback capped at 2 attempts. Evaluator budget guard unchanged. Perspective count capped at 3.
