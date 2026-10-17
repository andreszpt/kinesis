from __future__ import annotations

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
from hybrid_training_agent.config import get_chat_model
from hybrid_training_agent.tools.analytics_tools import ANALYTICS_TOOLS
from hybrid_training_agent.tools.planner_tools import PLANNER_TOOLS
from hybrid_training_agent.tools.science_tools import (
    SCIENCE_TOOLS,
    get_curated_hybrid_training_evidence,
)

AgentName = Literal["analyst", "researcher", "planner", "FINISH"]


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
) -> list[AnyMessage]:
    """Execute a specialist sub-agent with its bound tools in a ReAct loop."""
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    working_messages: list[AnyMessage] = [
        SystemMessage(content=system_prompt),
        *_prepare_subagent_history(messages),
    ]
    new_messages: list[AnyMessage] = []

    for _ in range(max_iterations):
        response: AIMessage = llm_with_tools.invoke(working_messages)
        working_messages.append(response)
        new_messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {}) or {}
            tool_id = tc.get("id", tool_name)
            target_tool = tool_map.get(tool_name)
            if target_tool is None:
                obs_content = f'{{"error": "Unknown tool {tool_name}"}}'
            else:
                try:
                    result = target_tool.invoke(tool_args)
                    obs_content = (
                        result
                        if isinstance(result, str)
                        else __import__("json").dumps(result, ensure_ascii=False, default=str)
                    )
                except Exception as exc:
                    obs_content = __import__("json").dumps({"error": str(exc)})

            tool_msg = ToolMessage(
                content=obs_content,
                tool_call_id=tool_id,
                name=tool_name,
            )
            working_messages.append(tool_msg)
            new_messages.append(tool_msg)

    # Tag the final AIMessage with the sub-agent name for clear provenance
    if new_messages and isinstance(new_messages[-1], AIMessage):
        final_ai = new_messages[-1]
        final_ai.name = agent_tag

    return new_messages


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
        last_user_msg = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_user_msg = str(msg.content)
                break

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
        new_msgs = _run_tool_calling_subagent(
            llm=model,
            system_prompt=ANALYST_SYSTEM_PROMPT,
            tools=ANALYTICS_TOOLS,
            messages=state["messages"],
            agent_tag="Performance_and_Health_Analyst",
        )
        visited = list(state.get("visited_agents") or [])
        if "analyst" not in visited:
            visited.append("analyst")
        return {"messages": new_msgs, "visited_agents": visited}

    def researcher_node(state: HybridAgentState) -> dict[str, Any]:
        model = _get_llm()
        new_msgs = _run_tool_calling_subagent(
            llm=model,
            system_prompt=RESEARCHER_SYSTEM_PROMPT,
            tools=SCIENCE_TOOLS,
            messages=state["messages"],
            agent_tag="Sports_Science_Researcher",
        )
        visited = list(state.get("visited_agents") or [])
        if "researcher" not in visited:
            visited.append("researcher")
        return {"messages": new_msgs, "visited_agents": visited}

    def planner_node(state: HybridAgentState) -> dict[str, Any]:
        model = _get_llm()
        # Planner gets PLANNER_TOOLS + curated evidence lookup so it never lacks citations or syntax validation
        planner_toolkit = [*PLANNER_TOOLS, get_curated_hybrid_training_evidence]
        new_msgs = _run_tool_calling_subagent(
            llm=model,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            tools=planner_toolkit,
            messages=state["messages"],
            agent_tag="Hybrid_Coach_Planner",
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
