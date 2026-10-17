from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from pydantic import BaseModel, Field

from hybrid_training_agent.agents.prompts import (
    ANALYST_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
)
from hybrid_training_agent.config import get_chat_model, get_settings
from hybrid_training_agent.tools.analytics_tools import (
    PERFORMANCE_ANALYSIS_TOOLS,
    analyze_training_and_wellness,
    fetch_athlete_profile_and_zones,
)
from hybrid_training_agent.tools.planner_tools import (
    PLANNER_LIBRARY_TOOLS,
    delete_planned_event_from_intervals,
    get_upcoming_intervals_calendar,
    update_calendar_event,
    upload_training_plan_to_intervals,
    validate_and_preview_training_plan,
)
from hybrid_training_agent.tools.science_tools import (
    SCIENCE_TOOLS,
    get_curated_hybrid_training_evidence,
)

AgentName = Literal["analyst", "researcher", "planner", "FINISH"]

# Wording that means the athlete will manage the intervals.icu calendar, so the planner
# needs the delete tool bound (otherwise it is left out of the request payload).
_PLANNER_DELETE_KEYWORDS = ("delete", "remove", "cancel", "unplan", "clear the calendar")
_PLANNER_UPDATE_KEYWORDS = (
    "move",
    "reschedule",
    "shift",
    "postpone",
    "rename",
    "edit the",
    "change the",
    "update the",
    "swap",
)
_PLANNER_LIBRARY_KEYWORDS = (
    "library",
    "template",
    "folder",
    "save to",
    "save it",
    "save this",
    "apply plan",
    "apply the plan",
    "apply my plan",
    "from my plan",
    "reuse",
)
# Deep-dive analysis is opt-in so routine readiness checks stay a single model call.
_ANALYST_DEEP_KEYWORDS = (
    "analy",
    "performance",
    "benchmark",
    "curve",
    "best effort",
    "personal best",
    "pb ",
    "compare",
    "execution",
    "split",
    "stream",
    "decoupling",
    "drift",
    "efficiency",
    "when did",
    "similar session",
    "how did",
    "session review",
    "interval analysis",
    "each month",
    "every month",
    "month by month",
    "month-by-month",
    "monthly",
    "per month",
    "from 20",
    "since 20",
    "this year",
    "year to date",
    "year-to-date",
    "ytd",
    "breakdown",
)


class RouteDecision(BaseModel):
    """Supervisor routing decision for the next sub-agent to execute."""

    next_agent: AgentName = Field(
        ...,
        description="Choose 'analyst', 'researcher', 'planner', or 'FINISH' when the user's request is fully answered.",
    )
    reasoning: str = Field(
        default="", description="Brief rationale for selecting this sub-agent."
    )


class HybridAgentState(TypedDict):
    """Shared state across the Supervisor and Sub-Agents in LangGraph."""

    messages: Annotated[list[AnyMessage], add_messages]
    next_agent: str
    visited_agents: list[str]
    mode: str
    # Routing (tier 1 heuristic vs tier 2 LLM) for the current user turn.
    router: str
    pipeline: list[str]


@dataclass(frozen=True)
class RoutingPlan:
    """Tier-1 heuristic routing result for a single user turn."""

    pipeline: list[str]
    confident: bool
    reason: str


_THINKING_BLOCK_TYPES = ("thinking", "redacted_thinking")


def _extract_plain_text(content: Any) -> str:
    """Flatten a message content payload to plain text, dropping reasoning blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_type = block.get("type")
                if block_type in _THINKING_BLOCK_TYPES or block_type == "tool_use":
                    continue
                text = block.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _prepare_subagent_history(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Reduce shared state to a safe, provider-agnostic briefing for a sub-agent.

    Sub-agents run sequentially and each appends its full ReAct transcript (thinking
    blocks, tool calls, tool results) to the shared state. Replaying that transcript
    inside the next sub-agent's conversation breaks Anthropic validation: thinking
    block signatures are bound to the system prompt and tools of the conversation
    that created them ("Invalid `signature` in `thinking` block. The block is bound
    to a different conversation"). Consecutive assistant turns are also rejected as
    unsupported prefill.

    So we keep only the human turns and the plain-text conclusions of prior
    sub-agents, dropping reasoning/tool traffic, then guarantee the conversation ends
    with a user turn.
    """
    briefing: list[AnyMessage] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            briefing.append(HumanMessage(content=_extract_plain_text(msg.content)))
            continue
        if isinstance(msg, AIMessage):
            text = _extract_plain_text(msg.content).strip()
            if not text:
                continue
            if msg.name:
                text = f"{msg.name}:\n{text}"
            briefing.append(AIMessage(content=text))

    if briefing and isinstance(briefing[-1], AIMessage):
        briefing.append(
            HumanMessage(
                content=(
                    "Continue from the conversation above. Use the previous "
                    "sub-agent findings as context and complete your own part of "
                    "the task now. Call your tools as needed, then give your final answer."
                )
            )
        )
    return briefing


def _run_tool_calling_subagent(
    llm: BaseChatModel,
    system_prompt: str,
    tools: list[BaseTool],
    messages: list[AnyMessage],
    agent_tag: str,
    max_iterations: int = 6,
    context: str = "",
) -> list[AnyMessage]:
    """Execute a specialist sub-agent with its bound tools in a ReAct loop.

    `context` holds pre-fetched tool results that are injected into the system message so
    the model can answer without spending a round trip deciding which tool to call.
    """
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    system_content = system_prompt
    if context:
        system_content = (
            f"{system_prompt}\n\n"
            "# Pre-fetched tool results (already retrieved; use them directly and do not "
            f"re-run these tools)\n{context}"
        )

    working_messages: list[AnyMessage] = [
        SystemMessage(content=system_content),
        *_prepare_subagent_history(messages),
    ]
    new_messages: list[AnyMessage] = []

    for _ in range(max_iterations):
        response: AIMessage = llm_with_tools.invoke(working_messages)
        working_messages.append(response)
        new_messages.append(response)

        if not response.tool_calls:
            break

        # Independent tool calls (e.g. profile + analytics) are I/O bound, so run them
        # concurrently while preserving the model's original ordering.
        tool_messages = _execute_tool_calls(response.tool_calls, tool_map)
        working_messages.extend(tool_messages)
        new_messages.extend(tool_messages)

    # Tag the final AIMessage with the sub-agent name for clear provenance
    if new_messages and isinstance(new_messages[-1], AIMessage):
        final_ai = new_messages[-1]
        final_ai.name = agent_tag

    return new_messages


def _execute_tool_call(tool_call: dict[str, Any], tool_map: dict[str, BaseTool]) -> ToolMessage:
    """Invoke a single tool call and wrap the outcome in a ToolMessage."""
    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {}) or {}
    tool_id = tool_call.get("id", tool_name)
    target_tool = tool_map.get(tool_name)
    if target_tool is None:
        obs_content = json.dumps({"error": f"Unknown tool {tool_name}"})
    else:
        try:
            result = target_tool.invoke(tool_args)
            obs_content = (
                result
                if isinstance(result, str)
                else json.dumps(result, ensure_ascii=False, default=str)
            )
        except Exception as exc:  # noqa: BLE001 - surface tool failures to the model
            obs_content = json.dumps({"error": str(exc)})

    return ToolMessage(content=obs_content, tool_call_id=tool_id, name=tool_name)


def _execute_tool_calls(
    tool_calls: list[dict[str, Any]], tool_map: dict[str, BaseTool]
) -> list[ToolMessage]:
    if len(tool_calls) <= 1:
        return [_execute_tool_call(tc, tool_map) for tc in tool_calls]
    with ThreadPoolExecutor(max_workers=min(len(tool_calls), 8)) as pool:
        return list(pool.map(lambda tc: _execute_tool_call(tc, tool_map), tool_calls))


def _invoke_tool_specs(specs: list[tuple[BaseTool, dict[str, Any]]]) -> list[tuple[str, Any]]:
    """Run deterministic read-only tools concurrently, returning `(name, result)` pairs."""
    def run(spec: tuple[BaseTool, dict[str, Any]]) -> tuple[str, Any]:
        tool_obj, args = spec
        try:
            return tool_obj.name, tool_obj.invoke(args)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully to an error payload
            return tool_obj.name, {"error": str(exc)}

    if len(specs) <= 1:
        return [run(spec) for spec in specs]
    with ThreadPoolExecutor(max_workers=min(len(specs), 8)) as pool:
        return list(pool.map(run, specs))


def _format_tool_context(results: list[tuple[str, Any]]) -> str:
    """Render pre-fetched tool results as a compact JSON block for the system message."""
    blocks: list[str] = []
    for name, result in results:
        payload = (
            result
            if isinstance(result, str)
            else json.dumps(result, ensure_ascii=False, default=str)
        )
        blocks.append(f"### `{name}`\n```json\n{payload}\n```")
    return "\n\n".join(blocks)


def _latest_user_text(messages: list[AnyMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


_MONTH_NUMBERS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_UNIT_DAYS = {
    "d": 1, "day": 1, "days": 1,
    "w": 7, "week": 7, "weeks": 7,
    "mo": 30, "month": 30, "months": 30,
    "y": 365, "year": 365, "years": 365,
}
_MAX_WINDOW_DAYS = 3650


def _days_since(start: date) -> int:
    return max(7, min((date.today() - start).days + 1, _MAX_WINDOW_DAYS))


def _detect_days_back(user_text: str, default: int = 28) -> int:
    """Resolve the history window a request needs (in days).

    Handles explicit spans ("last 60 days", "past 3 months"), year references
    ("from 2026", "this year"), and month references ("since May"), so questions like
    "how many kms did I run each month from 2026" pre-fetch the whole period instead of
    the default 28 days.
    """
    lower = user_text.lower()

    span = re.search(
        r"(?:last|past|previous|trailing|final)\s+(\d+)\s*"
        r"(d|day|days|w|week|weeks|mo|month|months|y|year|years)\b",
        lower,
    )
    if span:
        return max(7, min(int(span.group(1)) * _UNIT_DAYS[span.group(2)], _MAX_WINDOW_DAYS))

    year_match = re.search(r"\b(?:from|since|in|during)\s+(20\d{2})\b", lower) or re.search(
        r"\b(20\d{2})\b", lower
    )
    if year_match:
        return _days_since(date(int(year_match.group(1)), 1, 1))

    if any(
        kw in lower
        for kw in (
            "this year",
            "year to date",
            "year-to-date",
            "ytd",
            "each month",
            "every month",
            "month by month",
            "month-by-month",
            "monthly",
            "per month",
        )
    ):
        return _days_since(date(date.today().year, 1, 1))

    month_match = re.search(r"(?:from|since)\s+([a-z]{3,9})", lower)
    if month_match and month_match.group(1) in _MONTH_NUMBERS:
        month = _MONTH_NUMBERS[month_match.group(1)]
        start = date(date.today().year, month, 1)
        if start > date.today():
            start = date(date.today().year - 1, month, 1)
        return _days_since(start)

    return default


def _select_analyst_tools(user_text: str) -> list[BaseTool]:
    """Bind the read-only benchmarking/execution tools only for analysis-style requests.

    Routine readiness checks keep an empty tool list so the analyst answers in one model
    call. Deep dives (session execution, PBs, curves, similar sessions) opt in explicitly.
    """
    lower = user_text.lower()
    if any(kw in lower for kw in _ANALYST_DEEP_KEYWORDS):
        return list(PERFORMANCE_ANALYSIS_TOOLS)
    return []


def _select_planner_tools(user_text: str) -> list[BaseTool]:
    """Bind only the planner tools the request can plausibly need.

    The intervals.icu calendar is pre-fetched, and the curated-evidence tool is only
    relevant when the researcher runs, so both are dropped from the request payload.
    Delete/update/library tools are bound solely for the requests that mention them.
    """
    tools: list[BaseTool] = [
        validate_and_preview_training_plan,
        upload_training_plan_to_intervals,
    ]
    lower = user_text.lower()
    if any(kw in lower for kw in _PLANNER_DELETE_KEYWORDS):
        tools.append(delete_planned_event_from_intervals)
    if any(kw in lower for kw in _PLANNER_UPDATE_KEYWORDS):
        tools.append(update_calendar_event)
    if any(kw in lower for kw in _PLANNER_LIBRARY_KEYWORDS):
        tools.extend(PLANNER_LIBRARY_TOOLS)
    return tools


# Intent keywords. `week`/`workout`/`run` are deliberately NOT plan triggers on their own:
# "how many kms did I run this week" is a data question, not a planning request.
_PLAN_KEYWORDS = (
    "plan",
    "schedule",
    "program",
    "mesocycle",
    "microcycle",
    "periodiz",
    "prescribe",
    "upload",
    "training block",
)
_PLAN_ACTION_VERBS = (
    "create",
    "make",
    "build",
    "design",
    "write",
    "generate",
    "prepare",
    "draft",
    "give me",
    "set up",
    "put together",
    "add to",
    "put on",
    "move",
    "reschedule",
    "shift",
    "postpone",
    "rename",
    "edit",
    "update",
    "change",
    "swap",
    "delete",
    "remove",
    "cancel",
    "save",
    "apply",
)
_PLAN_SUBJECT_NOUNS = (
    "workout",
    "session",
    "training",
    "run",
    "runs",
    "ride",
    "strength",
    "interval",
    "tempo",
    "fartlek",
    "long run",
    "easy run",
    "recovery run",
    "hybrid",
    "block",
)
_QUESTION_MARKERS = (
    "how many",
    "how much",
    "how far",
    "how long",
    "how often",
    "how did",
    "how is",
    "how are",
    "what was",
    "what's my",
    "whats my",
    "what is my",
    "what are my",
    "did i",
    "have i",
    "when did",
    "when was",
    "show me",
    "tell me",
    "summary",
    "stats",
)
_EVIDENCE_KEYWORDS = (
    "science",
    "scientific",
    "evidence",
    "citation",
    "citations",
    "cite",
    "literature",
    "peer-reviewed",
    "study",
    "studies",
    "paper",
    "pubmed",
    "pmid",
    "doi",
    "meta-analysis",
    "interference",
    "hypertrophy",
    "economy",
    "polarized",
    "why",
    "research",
)
_ANALYSIS_KEYWORDS = (
    "analy",
    "wellness",
    "hrv",
    "ctl",
    "atl",
    "tsb",
    "fatigue",
    "recovery",
    "readiness",
    "past",
    "history",
    "suunto",
    "log",
    "rpe",
    "efficiency",
    "data",
    "km",
    "kms",
    "kilomet",
    "mile",
    "weekly",
    "this week",
    "last week",
    "summary",
    "stats",
    *_QUESTION_MARKERS,
)


def classify_intent(user_text: str, mode: str = "auto") -> RoutingPlan:
    """Tier-1 heuristic routing for a user request.

    Planning requires explicit planning intent (a plan keyword, or an action verb aimed at a
    workout/training noun) so data questions like "how many kms did I run this week" go to the
    analyst only. The literature sub-agent is only added when the user explicitly asks for
    scientific evidence/citations.

    When nothing matches, the plan is flagged `confident=False` so the graph can fall back to
    the Tier-2 LLM supervisor instead of guessing.
    """
    if mode == "research":
        return RoutingPlan(["researcher"], True, "mode=research")
    if mode == "analyze":
        return RoutingPlan(["analyst"], True, "mode=analyze")

    lower = user_text.lower()
    is_question = any(marker in lower for marker in _QUESTION_MARKERS)
    has_plan_action = any(verb in lower for verb in _PLAN_ACTION_VERBS)
    wants_plan = mode == "plan" or any(kw in lower for kw in _PLAN_KEYWORDS) or (
        has_plan_action and any(noun in lower for noun in _PLAN_SUBJECT_NOUNS)
    )
    # A pure question ("what's my training plan?") is answered from data, not turned into a plan.
    if wants_plan and is_question and not has_plan_action:
        wants_plan = False

    wants_evidence = any(kw in lower for kw in _EVIDENCE_KEYWORDS)
    wants_analysis = any(kw in lower for kw in _ANALYSIS_KEYWORDS)

    if wants_plan:
        # Planning always inspects the athlete's data, but the literature sub-agent is
        # only added when the user explicitly asks for scientific evidence/citations.
        pipeline = ["analyst"]
        if wants_evidence:
            pipeline.append("researcher")
        pipeline.append("planner")
        return RoutingPlan(pipeline, True, "explicit planning intent")

    if wants_analysis and wants_evidence:
        return RoutingPlan(["analyst", "researcher"], True, "analysis + evidence")
    if wants_evidence:
        return RoutingPlan(["researcher"], True, "evidence request")
    if wants_analysis:
        return RoutingPlan(["analyst"], True, "data/analysis question")

    # No explicit signal (e.g. a short follow-up like "sounds good, go ahead"): not confident,
    # so the Tier-2 LLM supervisor can read the conversation and decide.
    return RoutingPlan(["analyst", "planner"], False, "no explicit intent signal")


def classify_intent_pipeline(user_text: str, mode: str = "auto") -> list[str]:
    """Backwards-compatible wrapper returning just the Tier-1 pipeline."""
    return classify_intent(user_text, mode=mode).pipeline


def build_hybrid_training_graph(
    llm: BaseChatModel | None = None,
    router_llm: BaseChatModel | None = None,
):
    """Build and compile the LangGraph Supervisor + 3 Sub-Agents state graph.

    Routing is three-tiered:
      1. Tier 1 - deterministic heuristic (`classify_intent`). Used when the intent is explicit.
      2. Tier 2 - structured-output LLM supervisor (`router_llm`, defaulting to `llm`), consulted
         only when Tier 1 is not confident (e.g. short follow-ups like "sounds good, go ahead").
      3. Tier 3 - the supervisor is a real per-step loop: it returns a `Command` for the next hop
         (one sub-agent at a time) instead of a fixed pipeline, so it can adapt mid-run.
    """
    def _get_llm() -> BaseChatModel:
        return llm if llm is not None else get_chat_model()

    def _get_router_llm() -> BaseChatModel | None:
        """Tier-2/3 router model: explicit override, KINESIS_ROUTER_MODEL, else the main model."""
        if router_llm is not None:
            return router_llm
        model_name = os.getenv("KINESIS_ROUTER_MODEL", "").strip()
        if model_name:
            return get_chat_model(model=model_name)
        if llm is not None:
            return llm
        return get_chat_model()

    def _next_hop(pipeline: list[str], visited: list[str]) -> str:
        for candidate in pipeline:
            if candidate not in visited:
                return candidate
        return "FINISH"

    def _target(agent: str) -> str:
        return END if agent in ("FINISH", "__end__") else agent

    def _llm_route(state: HybridAgentState, visited: list[str]) -> str | None:
        """Tier-2/Tier-3 LLM routing. Returns the next agent, or None on failure."""
        model = _get_router_llm()
        if model is None:
            return None
        try:
            structured = model.with_structured_output(RouteDecision)
            visited_hint = ", ".join(visited) if visited else "none"
            decision: RouteDecision = structured.invoke(
                [
                    SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
                    *_prepare_subagent_history(state["messages"]),
                    HumanMessage(
                        content=(
                            f"Sub-agents already run this turn: {visited_hint}. "
                            "Reply with the single next sub-agent to run, or FINISH."
                        )
                    ),
                ]
            )
            return decision.next_agent
        except Exception:  # noqa: BLE001 - routing must never crash the turn
            return None

    def supervisor_node(state: HybridAgentState) -> Command:
        visited = list(state.get("visited_agents") or [])
        mode = state.get("mode") or "auto"
        user_text = _latest_user_text(state["messages"])
        pipeline = list(state.get("pipeline") or [])
        router = state.get("router") or ""

        # First call of a turn: decide who should act, using Tier 1 then (if unsure) Tier 2.
        if not pipeline and not router:
            plan = classify_intent(user_text, mode=mode)
            if plan.confident:
                return Command(
                    goto=_target(_next_hop(plan.pipeline, visited)),
                    update={"pipeline": plan.pipeline, "router": "heuristic"},
                )
            decision = _llm_route(state, visited)
            if decision and decision not in visited:
                return Command(
                    goto=_target(decision),
                    update={"pipeline": plan.pipeline, "router": "llm"},
                )
            # Router unavailable/failed or repeated an agent: fall back to the heuristic plan.
            return Command(
                goto=_target(_next_hop(plan.pipeline, visited)),
                update={"pipeline": plan.pipeline, "router": "heuristic-fallback"},
            )

        # Tier-3 loop: an LLM-supervised turn re-decides after every hop.
        if router == "llm":
            decision = _llm_route(state, visited)
            if decision and decision not in visited:
                return Command(goto=_target(decision))
            return Command(goto=END)

        return Command(goto=_target(_next_hop(pipeline, visited)))

    def analyst_node(state: HybridAgentState) -> dict[str, Any]:
        model = _get_llm()
        user_text = _latest_user_text(state["messages"])
        # Deterministically pre-fetch the athlete data instead of spending an LLM round
        # trip on tool selection. Runs with no bound tools so it always answers in one call.
        if get_settings().has_intervals_credentials:
            days_back = _detect_days_back(user_text)
            results = _invoke_tool_specs(
                [
                    (fetch_athlete_profile_and_zones, {}),
                    (analyze_training_and_wellness, {"days_back": days_back}),
                ]
            )
            context = _format_tool_context(results)
        else:
            context = (
                "intervals.icu credentials are not configured, so no athlete data is "
                "available. State that data could not be retrieved."
            )
        new_msgs = _run_tool_calling_subagent(
            llm=model,
            system_prompt=ANALYST_SYSTEM_PROMPT,
            tools=_select_analyst_tools(user_text),
            messages=state["messages"],
            agent_tag="Performance_and_Health_Analyst",
            context=context,
        )
        visited = list(state.get("visited_agents") or [])
        if "analyst" not in visited:
            visited.append("analyst")
        return {"messages": new_msgs, "visited_agents": visited}

    def researcher_node(state: HybridAgentState) -> dict[str, Any]:
        model = _get_llm()
        # The curated evidence matrix is local and instant: inject it so the researcher only
        # spends a round trip on a live PubMed/Europe PMC search when it truly needs one.
        curated = get_curated_hybrid_training_evidence.invoke({"topic": "all"})
        context = _format_tool_context([("get_curated_hybrid_training_evidence", curated)])
        new_msgs = _run_tool_calling_subagent(
            llm=model,
            system_prompt=RESEARCHER_SYSTEM_PROMPT,
            tools=SCIENCE_TOOLS,
            messages=state["messages"],
            agent_tag="Sports_Science_Researcher",
            context=context,
        )
        visited = list(state.get("visited_agents") or [])
        if "researcher" not in visited:
            visited.append("researcher")
        return {"messages": new_msgs, "visited_agents": visited}

    def planner_node(state: HybridAgentState) -> dict[str, Any]:
        model = _get_llm()
        user_text = _latest_user_text(state["messages"])
        # Pre-fetch the current calendar so the planner does not need a selection round trip,
        # and bind only the tools the request can actually use.
        if get_settings().has_intervals_credentials:
            results = _invoke_tool_specs([(get_upcoming_intervals_calendar, {"days_ahead": 14})])
            context = _format_tool_context(results)
        else:
            context = (
                "intervals.icu credentials are not configured, so the calendar could not be "
                "read. Do not attempt to upload; provide the plan for review only."
            )
        new_msgs = _run_tool_calling_subagent(
            llm=model,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            tools=_select_planner_tools(user_text),
            messages=state["messages"],
            agent_tag="Hybrid_Coach_Planner",
            context=context,
        )
        visited = list(state.get("visited_agents") or [])
        if "planner" not in visited:
            visited.append("planner")
        return {"messages": new_msgs, "visited_agents": visited}

    workflow = StateGraph(HybridAgentState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("planner", planner_node)

    workflow.add_edge(START, "supervisor")
    # The supervisor routes per step via `Command(goto=...)` (Tier 3), so there are no
    # conditional edges from it; each sub-agent returns to the supervisor to continue.
    workflow.add_edge("analyst", "supervisor")
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("planner", "supervisor")

    return workflow.compile()
