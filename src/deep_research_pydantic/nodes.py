"""Pydantic AI node implementations for the deep research pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from deep_research.config import MCPServerConfig
from deep_research.evidence_factory import (
    build_evidence_from_tool_result,
    build_tool_call_record,
)
from deep_research.mcp_client import MCPClientManager
from deep_research.models import (
    CompressedFindings,
    EvaluatorDecision,
    Evidence,
    SubQuestion,
    ToolCall,
    VerificationResult,
)
from deep_research.prompts import (
    CLARIFIER_SYSTEM,
    COMPRESSOR_SYSTEM,
    EVALUATOR_SYSTEM_V2,
    PERSPECTIVE_SYSTEM,
    PLANNER_SYSTEM,
    QUERY_ADAPTER_SYSTEM,
    SCORER_SYSTEM,
    SYNTHESIZER_CHAT_SYSTEM,
    SYNTHESIZER_REPORT_SYSTEM,
    VERIFIER_SYSTEM,
)
from deep_research_pydantic.agents import AgentBundle
from deep_research_pydantic.state import ResearchState

logger = logging.getLogger(__name__)

_STRUCTURED_OUTPUT_ERRORS = (
    UnexpectedModelBehavior,
    ModelRetry,
    ValidationError,
)

_MAX_HISTORY_TURNS = 10

_RISK_TIERS: list[tuple[str, str]] = [
    ("genie", "safe"),
    ("vector_search", "safe"),
    ("vs", "safe"),
    ("knowledge", "restricted"),
    ("ka", "restricted"),
]

_TOOL_TYPE_GUIDANCE = [
    (
        "genie",
        "data_query",
        "Rephrase as a natural-language data question suitable for SQL analysis",
    ),
    (
        "vector_search",
        "semantic_search",
        "Extract key terms and rephrase as a semantic search query",
    ),
    (
        "vs",
        "semantic_search",
        "Extract key terms and rephrase as a semantic search query",
    ),
    (
        "knowledge",
        "knowledge_retrieval",
        "Rephrase as a focused knowledge retrieval question",
    ),
    (
        "ka",
        "knowledge_retrieval",
        "Rephrase as a focused knowledge retrieval question",
    ),
]


@dataclass
class NodeDeps:
    """Runtime dependencies shared by all nodes."""

    agents: AgentBundle
    mcp_manager: MCPClientManager | None
    job_manager: Any | None = None
    job_id: str = ""


def to_message_history(
    conversation_history: list[dict[str, Any]],
    limit: int = _MAX_HISTORY_TURNS,
) -> list[ModelMessage]:
    """Convert API conversation entries into Pydantic AI model messages."""

    history: list[ModelMessage] = []
    messages = conversation_history[-limit:] if limit > 0 else []

    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if role == "user":
            history.append(ModelRequest(parts=[UserPromptPart(content)]))
        elif role == "assistant":
            history.append(ModelResponse(parts=[TextPart(content)]))

    return history


def _normalize_question(question: str) -> str:
    return " ".join(question.lower().strip().split())


def _merge_plans(
    existing: list[SubQuestion],
    new_items: list[SubQuestion],
) -> list[SubQuestion]:
    existing_questions = {
        _normalize_question(sub_question.question)
        for sub_question in existing
    }
    merged = list(existing)

    for sub_question in new_items:
        normalized = _normalize_question(sub_question.question)
        if normalized not in existing_questions:
            merged.append(sub_question)
            existing_questions.add(normalized)
        else:
            logger.debug(
                "Skipping duplicate sub-question: %s",
                sub_question.question[:60],
            )

    return merged


def _build_tool_catalog(
    tools: list[str],
    mcp_manager: MCPClientManager | None = None,
) -> str:
    if not mcp_manager:
        return "Available tools: " + ", ".join(sorted(tools))

    available: dict[str, MCPServerConfig] = mcp_manager.get_available_servers()
    lines = ["Available tools:"]

    for tool_name in sorted(tools):
        config = available.get(tool_name)
        if config:
            tool_type = config.managed_type or "general"
            capability = config.capability or "general"
            display_name = config.display_name or config.name
            description = config.description or "No description available."
            lines.append(
                f"- {config.name} [{tool_type}, {capability}]: "
                f'"{display_name}" — {description}'
            )
        else:
            lines.append(f"- {tool_name}: (no description available)")

    return "\n".join(lines)


def _get_risk_tier(tool_name: str) -> str:
    lower = tool_name.lower()
    for keyword, tier in _RISK_TIERS:
        if keyword in lower:
            return tier
    return "safe"


def _tool_error_rate(tool_name: str, tool_call_log: list[ToolCall]) -> float:
    calls = [
        tool_call
        for tool_call in tool_call_log
        if tool_call.tool_name == tool_name
    ]
    if not calls:
        return 0.0

    errors = sum(1 for tool_call in calls if tool_call.status == "error")
    return errors / len(calls)


def _classify_tool(tool_name: str) -> tuple[str, str]:
    lower = tool_name.lower()
    for keyword, tool_type, guidance in _TOOL_TYPE_GUIDANCE:
        if keyword in lower:
            return tool_type, guidance
    return "default", ""


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[*_`#>]", "", text)
    return text


def _compute_content_hash(source_id: str, snippet: str) -> str:
    normalized_source = source_id.lower().strip()
    normalized_snippet = _normalize_text(snippet)
    content = f"{normalized_source}\n{normalized_snippet}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _fallback_sub_question(
    query: str,
    tools: list[str],
    iteration: int,
) -> SubQuestion:
    return SubQuestion(
        subquestion_id=f"sq-{uuid.uuid4().hex[:8]}",
        question=query,
        assigned_tools=tools,
        iteration_created=iteration,
    )


def _add_usage(
    prior: dict[str, int],
    input_tokens: int,
    output_tokens: int,
) -> dict[str, int]:
    return {
        "input": int(prior.get("input", 0)) + int(input_tokens),
        "output": int(prior.get("output", 0)) + int(output_tokens),
    }


async def _push_job_event(
    job_manager: Any,
    job_id: str,
    event: dict[str, Any],
) -> None:
    result = job_manager.push_event(job_id, event)
    if inspect.isawaitable(result):
        await result


async def clarifier(
    state: ResearchState,
    deps: NodeDeps,
) -> dict[str, Any]:
    """Refine the query or request clarification for genuine ambiguity."""

    if state.get("clarified_query"):
        return {"clarified_query": state["clarified_query"]}

    history = to_message_history(
        state.get("conversation_history", []),
        limit=_MAX_HISTORY_TURNS,
    )

    try:
        result = await deps.agents.clarifier.run(
            state["user_query"],
            message_history=history,
            instructions=CLARIFIER_SYSTEM,
        )
        output = result.output
    except _STRUCTURED_OUTPUT_ERRORS:
        logger.warning(
            "Clarifier structured output validation failed; using original query",
            exc_info=True,
        )
        return {"clarified_query": state["user_query"]}

    if output.needs_clarification:
        return {
            "clarified_query": output.best_guess or state["user_query"],
            "needs_clarification": True,
            "clarification_question": (
                output.question or "Could you clarify your question?"
            ),
            "clarification_options": output.options,
        }

    return {
        "clarified_query": output.clarified_query or state["user_query"],
    }


async def planner(
    state: ResearchState,
    deps: NodeDeps,
) -> dict[str, Any]:
    """Create or extend the research plan."""

    query = state.get("clarified_query") or state["user_query"]
    tools = state.get("selected_tools", [])
    iteration = state.get("iteration_count", 0)
    existing_plan = state.get("research_plan", [])
    missing_facets = state.get("missing_facets", [])
    existing_perspectives = state.get("perspectives", [])
    verification_result = state.get("verification_result")

    perspectives = existing_perspectives
    if not existing_plan and not perspectives:
        try:
            perspective_result = await deps.agents.perspective.run(
                query,
                instructions=PERSPECTIVE_SYSTEM,
            )
            perspectives = perspective_result.output.perspectives
        except _STRUCTURED_OUTPUT_ERRORS:
            logger.warning(
                "Perspective output validation failed; continuing without perspectives",
                exc_info=True,
            )
            perspectives = []

    prior_evidence = state.get("prior_evidence", [])
    prior_summary = ""
    if prior_evidence:
        topics = {
            evidence.title
            for evidence in prior_evidence
            if evidence.title
        }
        tools_used = {
            evidence.tool_that_produced_it
            for evidence in prior_evidence
            if evidence.tool_that_produced_it
        }
        prior_summary = (
            f"Prior research ({len(prior_evidence)} evidence items from previous turns):\n"
            f"- Topics covered: {', '.join(sorted(topics)[:10])}\n"
            f"- Tools used: {', '.join(sorted(tools_used))}\n"
            "Avoid duplicating already-covered topics unless the query asks "
            "for more depth or comparison."
        )

    user_content = query
    if existing_plan and (iteration > 0 or verification_result):
        answered = [
            sub_question
            for sub_question in existing_plan
            if sub_question.status == "answered"
        ]
        unanswered = [
            sub_question
            for sub_question in existing_plan
            if sub_question.status != "answered"
        ]
        context_parts = [f"Original query: {query}"]

        if answered:
            context_parts.append(
                "Already answered: "
                + ", ".join(
                    sub_question.question for sub_question in answered
                )
            )
        if unanswered:
            context_parts.append(
                "Still unanswered: "
                + ", ".join(
                    sub_question.question for sub_question in unanswered
                )
            )
        if missing_facets:
            context_parts.append(
                f"Gaps identified: {', '.join(missing_facets)}"
            )
        if (
            verification_result
            and verification_result.unsupported_claims
        ):
            context_parts.append(
                "Unsupported claims to address: "
                + ", ".join(verification_result.unsupported_claims)
            )
        if prior_summary:
            context_parts.append(prior_summary)

        context_parts.append(
            "Generate ONLY new sub-questions to fill the gaps."
        )
        user_content = "\n".join(context_parts)
    elif perspectives:
        perspective_text = "\n".join(
            f"- {perspective}" for perspective in perspectives
        )
        user_content = (
            f"Query: {query}\n\n"
            f"Research from these perspectives:\n{perspective_text}\n\n"
            f"{prior_summary}"
            "Generate sub-questions that cover the query from each perspective."
        )
    elif prior_summary:
        user_content = f"Query: {query}\n\n{prior_summary}"

    tool_catalog = _build_tool_catalog(tools, deps.mcp_manager)
    instructions = PLANNER_SYSTEM.format(tool_catalog=tool_catalog)

    try:
        planner_result = await deps.agents.planner.run(
            user_content,
            instructions=instructions,
        )
        output = planner_result.output
    except _STRUCTURED_OUTPUT_ERRORS:
        logger.warning(
            "Planner structured output validation failed; using fallback plan",
            exc_info=True,
        )
        if not existing_plan:
            return {
                "research_plan": [
                    _fallback_sub_question(query, tools, iteration)
                ],
                "perspectives": perspectives,
            }
        return {
            "research_plan": existing_plan,
            "perspectives": perspectives,
        }

    allowed_tools = set(tools)
    new_sub_questions: list[SubQuestion] = []

    for planned in output.sub_questions:
        raw_tools = planned.assigned_tools
        valid_tools = [
            tool_name
            for tool_name in raw_tools
            if tool_name in allowed_tools
        ]

        if not valid_tools:
            valid_tools = list(tools)
            if raw_tools != tools:
                logger.warning(
                    "Planner assigned unknown tools %s, falling back to %s",
                    raw_tools,
                    valid_tools,
                )

        new_sub_questions.append(
            SubQuestion(
                subquestion_id=f"sq-{uuid.uuid4().hex[:8]}",
                question=planned.question or query,
                assigned_tools=valid_tools,
                iteration_created=iteration,
            )
        )

    merged = _merge_plans(existing_plan, new_sub_questions)
    return {
        "research_plan": merged,
        "perspectives": perspectives,
    }


async def authorizer(
    state: ResearchState,
    deps: NodeDeps,
) -> dict[str, Any]:
    """Validate planned tool access by risk tier and prior error rate."""

    del deps

    plan = state.get("research_plan", [])
    tool_call_log = state.get("tool_call_log", [])
    tool_assignments: dict[str, list[str]] = {}

    for sub_question in plan:
        approved_tools: list[str] = []

        for tool_name in sub_question.assigned_tools:
            tier = _get_risk_tier(tool_name)

            if tier == "safe":
                approved_tools.append(tool_name)
            elif tier == "restricted":
                logger.info("Restricted tool access: %s", tool_name)
                approved_tools.append(tool_name)
            elif tier == "privileged":
                if _tool_error_rate(tool_name, tool_call_log) > 0.5:
                    logger.warning(
                        "Downgrading %s — high error rate",
                        tool_name,
                    )
                    continue
                approved_tools.append(tool_name)
            else:
                approved_tools.append(tool_name)

            if tool_name not in tool_assignments:
                tool_assignments[tool_name] = []
            tool_assignments[tool_name].append(
                sub_question.subquestion_id
            )

        sub_question.assigned_tools = approved_tools

    return {
        "research_plan": plan,
        "tool_assignments": tool_assignments,
    }


async def query_adapter(
    state: ResearchState,
    deps: NodeDeps,
) -> dict[str, Any]:
    """Reformulate each unanswered sub-question for its assigned tools."""

    plan = state.get("research_plan", [])

    for sub_question in plan:
        if sub_question.status == "answered":
            continue

        for tool_name in sub_question.assigned_tools:
            tool_type, guidance = _classify_tool(tool_name)

            if tool_type == "default":
                adapted = sub_question.question
            else:
                instructions = QUERY_ADAPTER_SYSTEM.format(
                    tool_type=tool_type,
                    guidance=guidance,
                )
                result = await deps.agents.query_adapter.run(
                    sub_question.question,
                    instructions=instructions,
                )
                adapted = result.output.strip()

            sub_question.adapted_queries[tool_name] = adapted

    return {"research_plan": plan}


async def researcher(
    state: ResearchState,
    deps: NodeDeps,
) -> dict[str, Any]:
    """Execute sub-questions concurrently and tools within each sequentially."""

    plan = state.get("research_plan", [])
    iteration = state.get("iteration_count", 0) + 1
    existing_evidence = list(state.get("evidence", []))
    existing_tool_calls = list(state.get("tool_call_log", []))
    tool_calls_used = state.get("tool_calls_used", 0)
    budget = state.get("budget")

    pending = [
        sub_question
        for sub_question in plan
        if sub_question.status != "answered"
    ]

    if not pending:
        return {
            "evidence": existing_evidence,
            "tool_call_log": existing_tool_calls,
            "iteration_count": iteration,
            "tool_calls_used": tool_calls_used,
            "research_plan": plan,
        }

    remaining_budget = (
        budget.max_tool_calls - tool_calls_used
        if budget
        else 999
    )
    per_sub_question_budget = max(
        remaining_budget // len(pending),
        1,
    )

    async def research_one(
        sub_question: SubQuestion,
    ) -> tuple[list[Evidence], list[ToolCall], int]:
        new_evidence: list[Evidence] = []
        new_tool_calls: list[ToolCall] = []
        local_calls = 0

        for tool_name in sub_question.assigned_tools:
            if local_calls >= per_sub_question_budget:
                logger.warning("Per-sub-question budget cap reached")
                break

            query = sub_question.adapted_queries.get(
                tool_name,
                sub_question.question,
            )
            start_time = time.monotonic()
            started_at = datetime.now(timezone.utc)
            input_data = {"query": query}

            try:
                if deps.mcp_manager is None:
                    raise RuntimeError("MCP manager is not configured")

                result = await deps.mcp_manager.call_tool(
                    tool_name,
                    "query",
                    input_data,
                )
                elapsed_ms = int(
                    (time.monotonic() - start_time) * 1000
                )

                tool_call = build_tool_call_record(
                    tool_name=tool_name,
                    server_name=tool_name,
                    input_data=input_data,
                    output_data=result,
                    latency_ms=elapsed_ms,
                    iteration=iteration,
                    started_at=started_at,
                )
                new_tool_calls.append(tool_call)
                local_calls += 1

                evidences = build_evidence_from_tool_result(
                    tool_name=tool_name,
                    tool_call_id=tool_call.tool_call_id,
                    result=result,
                    iteration=iteration,
                    started_at=started_at,
                )
                new_evidence.extend(evidences)

                for evidence_item in evidences:
                    sub_question.evidence_ids.append(
                        evidence_item.evidence_id
                    )

                sub_question.status = "in_progress"
            except Exception as exc:
                elapsed_ms = int(
                    (time.monotonic() - start_time) * 1000
                )
                tool_call = build_tool_call_record(
                    tool_name=tool_name,
                    server_name=tool_name,
                    input_data=input_data,
                    output_data={},
                    latency_ms=elapsed_ms,
                    iteration=iteration,
                    started_at=started_at,
                    status="error",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                new_tool_calls.append(tool_call)
                local_calls += 1
                logger.warning(
                    "Tool call failed: %s - %s",
                    tool_name,
                    exc,
                )

        return new_evidence, new_tool_calls, local_calls

    results = await asyncio.gather(
        *[
            research_one(sub_question)
            for sub_question in pending
        ],
        return_exceptions=True,
    )

    all_evidence = list(existing_evidence)
    all_tool_calls = list(existing_tool_calls)
    total_calls = tool_calls_used

    for index, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.error(
                "Sub-question research failed: %s - %s",
                pending[index].question[:60],
                result,
            )
            continue

        evidence_items, tool_calls, calls = result
        all_evidence.extend(evidence_items)
        all_tool_calls.extend(tool_calls)
        total_calls += calls

    return {
        "evidence": all_evidence,
        "tool_call_log": all_tool_calls,
        "iteration_count": iteration,
        "tool_calls_used": total_calls,
        "research_plan": plan,
    }


async def normalizer(
    state: ResearchState,
    deps: NodeDeps,
) -> dict[str, Any]:
    """Normalize and deduplicate evidence."""

    del deps

    evidence = state.get("evidence", [])
    if not evidence:
        return {}

    seen_hashes: set[str] = set()
    deduplicated: list[Evidence] = []

    for evidence_item in evidence:
        if not evidence_item.content_hash:
            evidence_item.content_hash = _compute_content_hash(
                evidence_item.source_id,
                evidence_item.snippet,
            )

        if evidence_item.content_hash not in seen_hashes:
            seen_hashes.add(evidence_item.content_hash)
            deduplicated.append(evidence_item)
        else:
            logger.debug(
                "Dedup: skipping duplicate evidence %s",
                evidence_item.evidence_id,
            )

    return {"evidence": deduplicated}


async def scorer(
    state: ResearchState,
    deps: NodeDeps,
) -> dict[str, Any]:
    """Score evidence relevance and remove items below the threshold."""

    plan = state.get("research_plan", [])
    evidence = list(state.get("evidence", []))

    for sub_question in plan:
        sub_question_evidence = [
            evidence_item
            for evidence_item in evidence
            if evidence_item.evidence_id in sub_question.evidence_ids
        ]

        for evidence_item in sub_question_evidence:
            instructions = SCORER_SYSTEM.format(
                question=sub_question.question
            )
            try:
                score_result = await deps.agents.scorer.run(
                    evidence_item.snippet,
                    instructions=instructions,
                )
                score = score_result.output.score
            except _STRUCTURED_OUTPUT_ERRORS:
                logger.warning(
                    "Scorer structured output validation failed; "
                    "using moderate relevance",
                    exc_info=True,
                )
                score = 0.5

            evidence_item.confidence = score

    filtered = [
        evidence_item
        for evidence_item in evidence
        if evidence_item.confidence >= 0.3
    ]
    logger.info(
        "Scorer: %d items → %d after filtering (threshold=0.3)",
        len(evidence),
        len(filtered),
    )

    return {"evidence": filtered}


async def evaluator(
    state: ResearchState,
    deps: NodeDeps,
) -> dict[str, Any]:
    """Evaluate evidence sufficiency with per-sub-question verdicts."""

    plan = state.get("research_plan", [])
    evidence = state.get("evidence", [])
    prior_evidence = state.get("prior_evidence", [])
    iteration = state.get("iteration_count", 0)
    budget = state.get("budget")
    max_iterations = budget.max_iterations if budget else 5
    total_evidence_count = len(evidence) + len(prior_evidence)

    sub_question_details: list[str] = []
    for sub_question in plan:
        sub_question_evidence = [
            evidence_item
            for evidence_item in evidence
            if evidence_item.evidence_id in sub_question.evidence_ids
        ]
        evidence_text = "\n".join(
            f"  - [{evidence_item.source_id}] "
            f"(relevance={evidence_item.confidence:.2f}) "
            f"{evidence_item.snippet}"
            for evidence_item in sub_question_evidence
        )
        sub_question_details.append(
            f"Sub-question [{sub_question.subquestion_id}]: "
            f"{sub_question.question}\n"
            f"  Status: {sub_question.status}\n"
            f"  Evidence ({len(sub_question_evidence)} items):\n"
            f"{evidence_text or '  (none)'}"
        )

    instructions = EVALUATOR_SYSTEM_V2.format(
        sub_question_details="\n\n".join(sub_question_details),
        evidence_count=total_evidence_count,
        iteration=iteration,
        max_iterations=max_iterations,
    )

    try:
        evaluator_result = await deps.agents.evaluator.run(
            "Evaluate the evidence and provide your assessment.",
            instructions=instructions,
        )
        output = evaluator_result.output
    except _STRUCTURED_OUTPUT_ERRORS:
        logger.warning(
            "Evaluator structured output validation failed; stopping",
            exc_info=True,
        )
        fallback = EvaluatorDecision(
            sufficiency_score=0.5,
            missing_facets=[],
            recommended_actions=[],
            decision="stop",
            reason="Failed to parse evaluator response — stopping",
        )
        return {
            "evaluator_decision": fallback,
            "sufficiency_score": 0.5,
        }

    budget_exhausted = iteration >= max_iterations
    decision = output.decision

    if budget_exhausted and decision == "continue":
        decision = "stop"

    if decision == "continue" and iteration > 0:
        new_evidence = [
            evidence_item
            for evidence_item in evidence
            if evidence_item.iteration == iteration
        ]
        if not new_evidence:
            decision = "stop"
            logger.info(
                "No new evidence gathered — stopping research loop"
            )

    evaluator_decision = EvaluatorDecision(
        sufficiency_score=output.sufficiency_score,
        missing_facets=output.missing_facets,
        recommended_actions=output.recommended_actions,
        decision=decision,
        reason=output.reason,
        budget_exhausted=budget_exhausted,
    )

    single_source_sub_questions: list[str] = []
    for sub_question in plan:
        sub_question_evidence = [
            evidence_item
            for evidence_item in evidence
            if evidence_item.evidence_id in sub_question.evidence_ids
        ]
        unique_tools = {
            evidence_item.tool_that_produced_it
            for evidence_item in sub_question_evidence
        }

        if sub_question_evidence and len(unique_tools) <= 1:
            tool_name = (
                next(iter(unique_tools))
                if unique_tools
                else "unknown"
            )
            single_source_sub_questions.append(
                f"{sub_question.subquestion_id} relies only on {tool_name}"
            )
            logger.info(
                "Single source: %s uses only %s",
                sub_question.subquestion_id,
                unique_tools,
            )

    if single_source_sub_questions:
        logger.info(
            "Source diversity warning: %d sub-question(s) rely on a single tool",
            len(single_source_sub_questions),
        )

    if output.source_diversity_score is not None:
        logger.info(
            "LLM source_diversity_score: %.2f",
            output.source_diversity_score,
        )
    if output.single_source_questions:
        logger.info(
            "LLM single_source_questions: %s",
            output.single_source_questions,
        )

    verdicts = {
        verdict.id: verdict
        for verdict in output.sub_question_verdicts
    }
    for sub_question in plan:
        verdict = verdicts.get(sub_question.subquestion_id)
        if verdict is None or verdict.verdict is None:
            continue

        if verdict.verdict == "unanswered":
            sub_question.status = "pending"
        else:
            sub_question.status = verdict.verdict

    return {
        "evaluator_decision": evaluator_decision,
        "sufficiency_score": evaluator_decision.sufficiency_score,
        "missing_facets": evaluator_decision.missing_facets,
        "research_plan": plan,
    }


async def compressor(
    state: ResearchState,
    deps: NodeDeps,
) -> dict[str, Any]:
    """Compress evidence into structured findings."""

    evidence = state.get("evidence", [])
    evidence_text = "\n".join(
        f"- [{evidence_item.source_id}] {evidence_item.snippet}"
        for evidence_item in evidence
    )

    try:
        compressor_result = await deps.agents.compressor.run(
            f"Evidence to compress:\n{evidence_text}",
            instructions=COMPRESSOR_SYSTEM,
        )
        output = compressor_result.output
    except _STRUCTURED_OUTPUT_ERRORS:
        logger.warning(
            "Compressor structured output validation failed; "
            "using evidence snippets",
            exc_info=True,
        )
        return {
            "compressed_findings": CompressedFindings(
                key_findings=[
                    evidence_item.snippet
                    for evidence_item in evidence[:5]
                ],
                open_questions=[],
                uncertainties=[],
                contradictions=[],
            )
        }

    return {
        "compressed_findings": CompressedFindings(
            key_findings=output.key_findings,
            open_questions=output.open_questions,
            uncertainties=output.uncertainties,
            contradictions=output.contradictions,
        )
    }


async def synthesizer(
    state: ResearchState,
    deps: NodeDeps,
) -> dict[str, Any]:
    """Produce the final answer, streaming token deltas when configured."""

    findings = state.get("compressed_findings")
    evidence = state.get("evidence", [])
    output_mode = state.get("output_mode", "chat")
    query = state.get("clarified_query") or state["user_query"]

    job_manager = state.get("_job_manager") or deps.job_manager
    job_id = state.get("_job_id") or deps.job_id

    instructions = (
        SYNTHESIZER_REPORT_SYSTEM
        if output_mode == "report"
        else SYNTHESIZER_CHAT_SYSTEM
    )

    context_parts: list[str] = []

    conversation_history = state.get("conversation_history", [])
    if conversation_history:
        context_parts.append("Conversation context (prior turns):")
        for message in conversation_history[-6:]:
            role = message.get("role", "user")
            content = message.get("content", "")
            if len(content) > 500:
                content = content[:500] + "..."
            context_parts.append(f"  {role}: {content}")
        context_parts.append(
            "(Use prior conversation as background context only; "
            "prioritize current query and evidence.)"
        )
        context_parts.append("")

    context_parts.append(f"User query: {query}")

    if findings:
        context_parts.append(
            f"Key findings: {', '.join(findings.key_findings)}"
        )
        if findings.open_questions:
            context_parts.append(
                f"Open questions: {', '.join(findings.open_questions)}"
            )
        if findings.uncertainties:
            context_parts.append(
                f"Uncertainties: {', '.join(findings.uncertainties)}"
            )
        if findings.contradictions:
            context_parts.append(
                f"Contradictions: {', '.join(findings.contradictions)}"
            )

    if evidence:
        context_parts.append(
            "Evidence (cite using [Source: Tool — description] format):"
        )
        for evidence_item in evidence:
            tool_label = (
                evidence_item.tool_that_produced_it or "Unknown"
            ).replace("]", ")")
            title_label = (
                evidence_item.title or evidence_item.source_id
            ).replace("]", ")")
            context_parts.append(
                f"- [Source: {tool_label} — {title_label}] "
                f"{evidence_item.snippet}"
            )

    user_prompt = "\n".join(context_parts)
    prior_usage = state.get(
        "_token_usage",
        {"input": 0, "output": 0},
    )

    if job_manager and job_id:
        chunks: list[str] = []
        async with deps.agents.synthesizer.run_stream(
            user_prompt,
            instructions=instructions,
        ) as stream:
            async for token in stream.stream_text(delta=True):
                if token:
                    chunks.append(token)
                    await _push_job_event(
                        job_manager,
                        job_id,
                        {"type": "token", "content": token},
                    )

            usage = stream.usage
            logger.debug(
                "Streaming synthesis usage: input=%d output=%d",
                usage.input_tokens,
                usage.output_tokens,
            )

        final_output = "".join(chunks)
        token_usage = prior_usage
    else:
        result = await deps.agents.synthesizer.run(
            user_prompt,
            instructions=instructions,
        )
        final_output = result.output
        usage = result.usage
        token_usage = _add_usage(
            prior_usage,
            usage.input_tokens,
            usage.output_tokens,
        )

    return {
        "final_output": final_output,
        "_token_usage": token_usage,
    }


async def verifier(
    state: ResearchState,
    deps: NodeDeps,
) -> dict[str, Any]:
    """Verify that claims in the final output are supported by evidence."""

    final_output = state.get("final_output", "")
    evidence = state.get("evidence", [])
    verification_attempts = (
        state.get("verification_attempts", 0) + 1
    )

    evidence_text = "\n".join(
        f"- [{evidence_item.source_id}] "
        f"(relevance={evidence_item.confidence:.2f}) "
        f"{evidence_item.snippet}"
        for evidence_item in evidence
    )
    user_prompt = (
        f"Draft response:\n{final_output}\n\n"
        f"Available evidence:\n{evidence_text}"
    )

    try:
        verifier_result = await deps.agents.verifier.run(
            user_prompt,
            instructions=VERIFIER_SYSTEM,
        )
        output = verifier_result.output
    except _STRUCTURED_OUTPUT_ERRORS:
        logger.warning(
            "Verifier structured output validation failed; "
            "using permissive fallback",
            exc_info=True,
        )
        return {
            "verification_result": VerificationResult(
                all_claims_supported=True,
                unsupported_claims=[],
                weakened_claims=[],
                contradictions_noted=[],
            ),
            "verification_attempts": verification_attempts,
        }

    return {
        "verification_result": VerificationResult(
            all_claims_supported=output.all_claims_supported,
            unsupported_claims=output.unsupported_claims,
            weakened_claims=output.weakened_claims,
            contradictions_noted=output.contradictions_noted,
        ),
        "verification_attempts": verification_attempts,
    }
