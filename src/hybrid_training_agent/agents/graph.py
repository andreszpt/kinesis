from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
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


def _detect_days_back(user_text: str, default: int = 28) -> int:
    """Honor an explicit window such as 'last 60 days' when prefetching analyst data."""
    match = re.search(r"(?:last|past|previous)\s+(\d+)\s*(?:days?|d)\b", user_text, re.IGNORECASE)
    if not match:
        return default
    return max(7, min(int(match.group(1)), 365))


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


def classify_intent_pipeline(user_text: str, mode: str = "auto") -> list[str]:
    """Determine the optimal ordered sequence of specialist sub-agents for a user request.

    The literature sub-agent is only added when the user explicitly asks for scientific
    evidence/citations, so routine plan requests stay fast and free of citation dumps.
    """
    if mode == "research":
        return ["researcher"]
    if mode == "analyze":
        return ["analyst"]

    lower = user_text.lower()
    wants_plan = mode == "plan" or any(
        kw in lower
        for kw in (
            "plan",
            "schedule",
            "program",
            "week",
            "mesocycle",
            "microcycle",
            "workout",
            "upload",
            "calendar",
            "prescribe",
        )
    )
    wants_evidence = any(
        kw in lower
        for kw in (
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
    )
    wants_analysis = any(
        kw in lower
        for kw in (
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
        )
    )

    if wants_plan:
        # Planning always inspects the athlete's data, but the literature sub-agent is
        # only added when the user explicitly asks for scientific evidence/citations.
        pipeline = ["analyst"]
        if wants_evidence:
            pipeline.append("researcher")
        pipeline.append("planner")
        return pipeline

    if wants_analysis and wants_evidence:
        return ["analyst", "researcher"]
    if wants_evidence:
        return ["researcher"]
    if wants_analysis:
        return ["analyst"]

    # No explicit keywords: inspect the data, then produce the answer.
    return ["analyst", "planner"]


def build_hybrid_training_graph(llm: BaseChatModel | None = None):
    """Build and compile the LangGraph Supervisor + 3 Sub-Agents state graph."""
    def _get_llm() -> BaseChatModel:
        return llm if llm is not None else get_chat_model()

    def supervisor_node(state: HybridAgentState) -> dict[str, Any]:
        visited = list(state.get("visited_agents") or [])
        mode = state.get("mode") or "auto"

        # Find the latest HumanMessage
        last_user_msg = _latest_user_text(state["messages"])

        pipeline = classify_intent_pipeline(last_user_msg, mode=mode)
        for candidate in pipeline:
            if candidate not in visited:
                return {"next_agent": candidate}

        # Optional LLM supervisor check if pipeline wasn't triggered
        if not visited:
            try:
                model = _get_llm()
                structured_sup = model.with_structured_output(RouteDecision)
                decision: RouteDecision = structured_sup.invoke(
                    [
                        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
                        *state["messages"],
                    ]
                )
                if decision.next_agent in ("analyst", "researcher", "planner"):
                    return {"next_agent": decision.next_agent}
            except Exception:
                return {"next_agent": "analyst"}

        return {"next_agent": "FINISH"}

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

    def route_from_supervisor(
        state: HybridAgentState,
    ) -> Literal["analyst", "researcher", "planner", "__end__"]:
        nxt = state.get("next_agent", "FINISH")
        if nxt in ("analyst", "researcher", "planner"):
            return nxt  # type: ignore[return-value]
        return "__end__"

    workflow = StateGraph(HybridAgentState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("planner", planner_node)

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "analyst": "analyst",
            "researcher": "researcher",
            "planner": "planner",
            "__end__": END,
        },
    )
    workflow.add_edge("analyst", "supervisor")
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("planner", "supervisor")

    return workflow.compile()
