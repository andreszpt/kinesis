from __future__ import annotations

import json
import sys
from typing import Any

import typer
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from hybrid_training_agent.agents.graph import build_hybrid_training_graph
from hybrid_training_agent.clients.intervals_client import (
    IntervalsIcuClient,
    IntervalsIcuError,
)
from hybrid_training_agent.clients.science_client import ScienceLiteratureClient
from hybrid_training_agent.config import get_settings
from hybrid_training_agent.tools.analytics_tools import analyze_training_and_wellness

# Ensure UTF-8 output on Windows terminals so Rich panels, tables, and Markdown render cleanly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

app = typer.Typer(
    name="kinesis",
    help="Kinesis: Evidence-Based Hybrid Training (Running + Strength) Multi-Agent Coach powered by LangGraph, PubMed, and Intervals.icu.",
    no_args_is_help=False,
)
console = Console()


def extract_message_text(content: Any) -> str:
    """Extract clean readable Markdown text from LangChain string or structured content blocks.

    Handles:
    - Plain strings (`str`)
    - Anthropic / OpenAI Responses content block lists (`[{"type": "text", "text": "..."}, {"type": "tool_use", ...}]`)
    - Single dict blocks (`{"type": "text", "text": "..."}`)
    """
    if content is None:
        return ""

    if isinstance(content, str):
        stripped = content.strip()
        # If a string was accidentally serialized as JSON content blocks, unpack it
        if (stripped.startswith("[{") and stripped.endswith("}]")) or (
            stripped.startswith('{"') and stripped.endswith("}")
        ):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, (list, dict)):
                    return extract_message_text(parsed)
            except Exception:
                pass
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                if block.strip():
                    parts.append(block)
            elif isinstance(block, dict):
                b_type = str(block.get("type") or "").lower()
                # Skip tool calls and internal thinking blocks
                if b_type in ("tool_use", "tool_call", "thinking", "reasoning", "input_json_delta"):
                    continue
                if "text" in block and isinstance(block["text"], str) and block["text"].strip():
                    parts.append(block["text"])
                elif "content" in block and isinstance(block["content"], str) and block["content"].strip():
                    parts.append(block["content"])
        return "\n\n".join(parts)

    if isinstance(content, dict):
        b_type = str(content.get("type") or "").lower()
        if b_type in ("tool_use", "tool_call", "thinking", "reasoning"):
            return ""
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        if "content" in content and isinstance(content["content"], str):
            return content["content"]
        return json.dumps(content, indent=2, ensure_ascii=False)

    return str(content)


def _render_agent_turn(messages: list[AnyMessage], start_idx: int) -> None:
    """Render newly added ToolMessages and AIMessages from the LangGraph execution."""
    for msg in messages[start_idx:]:
        if isinstance(msg, ToolMessage):
            console.print(
                f"  [dim cyan]⚙ Tool completed:[/dim cyan] [bold cyan]{msg.name}[/bold cyan]"
            )
        elif isinstance(msg, AIMessage):
            text = extract_message_text(msg.content).strip()
            if not text:
                continue

            # If this AIMessage is an intermediate step that also invokes tools, print a compact status line
            if getattr(msg, "tool_calls", None):
                first_line = text.splitlines()[0].strip()
                if first_line:
                    console.print(f"  [dim italic]💬 {first_line}[/dim italic]")
                continue

            agent_label = getattr(msg, "name", None) or "Kinesis"
            border = "green"
            if "Researcher" in agent_label:
                border = "magenta"
            elif "Analyst" in agent_label:
                border = "cyan"
            elif "Planner" in agent_label:
                border = "green"

            console.print(
                Panel(
                    Markdown(text),
                    title=f"[bold]{agent_label.replace('_', ' ')}[/bold]",
                    border_style=border,
                    padding=(1, 2),
                )
            )


def _pending_question(messages: list[AnyMessage]) -> str:
    """Return the last assistant answer if it appears to be asking the user a question."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            text = extract_message_text(msg.content).strip()
            return text if "?" in text else ""
        if isinstance(msg, HumanMessage):
            break
    return ""


@app.command("status")
def status_cmd() -> None:
    """Check configuration, LLM API keys, and Intervals.icu connection status."""
    settings = get_settings()
    table = Table(title="Kinesis — Hybrid Training Agent Configuration Status")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Status / Value", style="white")

    try:
        provider = settings.resolve_provider()
        has_key = (
            (provider in ("opencode-go", "opencode", "go") and bool(settings.opencode_go_api_key))
            or (provider in ("anthropic", "claude") and bool(settings.anthropic_api_key))
            or (provider in ("openai", "gpt") and bool(settings.openai_api_key))
            or (provider in ("google", "gemini", "google-genai") and bool(settings.google_api_key))
        )
        if has_key:
            table.add_row("LLM Provider", f"[green]Configured ({provider})[/green]")
        else:
            table.add_row(
                "LLM Provider",
                f"[yellow]{provider} selected, but API key is missing in .env (e.g., OPENCODE_GO_API_KEY)[/yellow]",
            )
    except ValueError as exc:
        table.add_row("LLM Provider", f"[yellow]{exc}[/yellow]")

    table.add_row(
        "LLM Model Override",
        settings.llm_model or "[dim](Using provider default)[/dim]",
    )
    table.add_row(
        "Intervals.icu API Key",
        "[green]Configured[/green]"
        if settings.has_intervals_credentials
        else "[yellow]Missing (set INTERVALS_ICU_API_KEY in .env)[/yellow]",
    )
    table.add_row("Intervals.icu Athlete ID", settings.intervals_athlete_id)

    console.print(table)

    if settings.has_intervals_credentials:
        client = IntervalsIcuClient(settings=settings)
        try:
            profile = client.get_athlete_profile()
            console.print(
                Panel(
                    f"[green]Connected to intervals.icu![/green]\n"
                    f"Athlete: [bold]{profile.get('name', 'Unknown')}[/bold] (ID: {profile.get('id')})\n"
                    f"Resting HR: {profile.get('icu_resting_hr', 'N/A')} bpm | Weight: {profile.get('icu_weight', 'N/A')} kg",
                    title="Intervals.icu Live Connection",
                    border_style="green",
                )
            )
        except IntervalsIcuError as exc:
            console.print(f"[red]Intervals.icu connection error:[/red] {exc}")


@app.command("analyze")
def analyze_cmd(
    days: int = typer.Option(28, "--days", "-d", help="Number of past days to analyze."),
    ai: bool = typer.Option(
        False,
        "--ai",
        help="Run the full LangGraph Performance & Health Analyst Sub-Agent commentary.",
    ),
) -> None:
    """Analyze past Running, Strength, and Wellness (HRV/RHR/Sleep/CTL/ATL/TSB) from intervals.icu."""
    if ai:
        ask_cmd(
            prompt=(
                f"Analyze my past {days} days of training and wellness data from intervals.icu. "
                "Evaluate my CTL/ATL/TSB, ACWR, HRV & RHR recovery trends, running zone polarization, "
                "and strength training frequency/spacing."
            ),
            mode="analyze",
        )
        return

    with console.status(f"[cyan]Analyzing last {days} days of training & wellness...[/cyan]"):
        report = analyze_training_and_wellness.invoke({"days_back": days})

    if "error" in report:
        console.print(
            Panel(
                f"[yellow]{report['error']}[/yellow]",
                title="Intervals.icu Credentials Needed",
                border_style="yellow",
            )
        )
        return

    lf = report.get("load_and_form", {})
    rw = report.get("recovery_and_wellness", {})
    rs = report.get("running_summary", {})
    ss = report.get("strength_summary", {})
    audit = report.get("concurrent_interference_audit", {})

    summary_md = (
        f"### Load & Form ({days}-Day Window)\n"
        f"- **Fitness (CTL)**: `{lf.get('ctl_fitness')}` | **Fatigue (ATL)**: `{lf.get('atl_fatigue')}` | **Form (TSB)**: `{lf.get('tsb_form')}`\n"
        f"- **ACWR**: `{lf.get('acwr_ratio')}` — *{lf.get('acwr_interpretation')}*\n\n"
        f"### Autonomic & Health Readiness\n"
        f"- **HRV (RMSSD)**: Latest `{rw.get('hrv', {}).get('latest_hrv_rmssd')}` | 7d Avg `{rw.get('hrv', {}).get('rolling_7d_mean')}` vs Baseline `{rw.get('hrv', {}).get('baseline_mean')}` (CV: `{rw.get('hrv', {}).get('rolling_7d_cv_pct')}%`)\n"
        f"- **Resting HR**: Latest `{rw.get('resting_hr', {}).get('latest_rhr_bpm')}` bpm | 7d Avg `{rw.get('resting_hr', {}).get('rolling_7d_mean_bpm')}` bpm\n"
        f"- **Sleep (7d Avg)**: `{rw.get('sleep_and_subjective', {}).get('rolling_7d_avg_hours')}` hrs/night\n\n"
        f"### Running Efficiency & Polarization\n"
        f"- **Sessions**: `{rs.get('total_sessions')}` (`{rs.get('sessions_per_week')}/wk`) | **Weekly Avg Distance**: `{rs.get('weekly_avg_distance_km')} km/wk`\n"
        f"- **3-Zone Distribution**: Low (Z1-Z2): `{rs.get('intensity_distribution_3_zone', {}).get('zone_1_2_low_pct')}%` | "
        f"Moderate (Z3): `{rs.get('intensity_distribution_3_zone', {}).get('zone_3_moderate_grey_zone_pct')}%` | "
        f"High (Z4-Z5): `{rs.get('intensity_distribution_3_zone', {}).get('zone_4_5_high_pct')}%`\n\n"
        f"### Strength & Concurrent Spacing\n"
        f"- **Strength Sessions**: `{ss.get('total_sessions')}` (`{ss.get('sessions_per_week')}/wk`) | "
        f"**Total Duration**: `{ss.get('total_duration_hours')} hrs`\n"
        f"- **Interference Warnings**: `{audit.get('warnings_count')}`"
    )
    if audit.get("warnings"):
        summary_md += "\n" + "\n".join(f"  - ⚠️ {w}" for w in audit["warnings"])

    console.print(
        Panel(
            Markdown(summary_md),
            title="Kinesis — Hybrid Training & Wellness Analysis",
            border_style="cyan",
        )
    )


@app.command("evidence")
def evidence_cmd(
    query: str = typer.Option(
        "",
        "--query",
        "-q",
        help="Live PubMed/Europe PMC search query (e.g., 'concurrent training running economy').",
    ),
    topic: str = typer.Option(
        "all",
        "--topic",
        "-t",
        help="Curated hybrid topic: concurrent_interference, running_economy_and_strength, intensity_distribution_and_polarization, hrv_and_load_monitoring, or all.",
    ),
) -> None:
    """Search PubMed / Europe PMC or view curated peer-reviewed evidence for Hybrid Training."""
    client = ScienceLiteratureClient()
    if query.strip():
        with console.status(f"[magenta]Searching PubMed & Europe PMC for '{query}'...[/magenta]"):
            papers = client.search_pubmed(query=query, max_results=5)
            if not papers:
                papers = client.search_europe_pmc(query=query, max_results=5)

        if not papers:
            console.print(f"[yellow]No peer-reviewed papers found for query: {query}[/yellow]")
            return

        for idx, p in enumerate(papers, start=1):
            md = (
                f"**{idx}. {p.get('title')}**\n"
                f"- **Authors**: {p.get('authors')} ({p.get('year')}) — *{p.get('journal')}*\n"
                f"- **PMID**: `{p.get('pmid')}` | **DOI**: `{p.get('doi')}` | **URL**: {p.get('url')}\n\n"
                f"{p.get('abstract', '')[:800]}..."
            )
            console.print(Panel(Markdown(md), border_style="magenta"))
        return

    curated = client.get_curated_evidence(topic)
    for cat, citations in curated.items():
        lines = [f"### Category: `{cat}`\n"]
        for c in citations:
            lines.append(
                f"- **{c['authors']} ({c['year']})** — *{c['title']}* ({c['journal']})\n"
                f"  - **Study Type**: {c['study_type']} | **PMID**: `{c['pmid']}` | **DOI**: `{c['doi']}`\n"
                f"  - **Key Findings**: {c['key_findings']}\n"
                f"  - **Hybrid Programming Rule**: {c['practical_application']}\n"
            )
        console.print(
            Panel(
                Markdown("\n".join(lines)),
                title=f"Curated Sports Science Evidence: {cat}",
                border_style="magenta",
            )
        )


@app.command("ask")
def ask_cmd(
    prompt: str = typer.Argument(..., help="Your question or training plan request."),
    mode: str = typer.Option(
        "auto",
        "--mode",
        "-m",
        help="Execution pipeline mode: 'auto', 'plan', 'analyze', or 'research'.",
    ),
) -> None:
    """Run a one-shot query or plan generation through the LangGraph multi-agent system."""
    graph = build_hybrid_training_graph()
    initial_state = {
        "messages": [HumanMessage(content=prompt)],
        "next_agent": "supervisor",
        "visited_agents": [],
        "mode": mode,
    }
    with console.status("[bold green]Running Kinesis LangGraph Sub-Agents...[/bold green]"):
        final_state = graph.invoke(initial_state)

    _render_agent_turn(final_state["messages"], start_idx=1)

    # The agents may end a turn by asking the athlete a clarifying question (e.g. available
    # equipment, weekly frequency, or whether to upload). Offer a follow-up reply so the
    # one-shot command does not dead-end. Skipped automatically for non-interactive usage
    # (pipes, scripts, CI), which keeps stdout clean.
    if not sys.stdin.isatty():
        return

    conversation_messages = list(final_state["messages"])
    pending = _pending_question(conversation_messages)
    if pending:
        console.print(
            "[dim]Kinesis is waiting on your reply. "
            "Type your answer, or press Enter (or type 'done') to exit.[/dim]"
        )

    while pending:
        try:
            follow_up = Prompt.ask("\n[bold green]You[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Done.[/dim]")
            break

        if not follow_up or follow_up.lower() in ("done", "exit", "quit", "/quit", "/exit"):
            break

        conversation_messages.append(HumanMessage(content=follow_up))
        prev_len = len(conversation_messages)
        try:
            with console.status("[bold green]Continuing...[/bold green]"):
                state_out = graph.invoke(
                    {
                        "messages": conversation_messages,
                        "next_agent": "supervisor",
                        "visited_agents": [],
                        "mode": mode,
                    }
                )
        except Exception as exc:
            console.print(f"[red]Agent Execution Error:[/red] {exc}")
            break

        conversation_messages = list(state_out["messages"])
        _render_agent_turn(conversation_messages, start_idx=prev_len)
        pending = _pending_question(conversation_messages)


def _execute_graph_turn(
    graph: Any,
    conversation_messages: list[AnyMessage],
    mode: str,
    status_text: str,
) -> list[AnyMessage]:
    """Invoke the graph for a single turn, render new messages, and return the history."""
    prev_len = len(conversation_messages)
    try:
        with console.status(status_text):
            state_out = graph.invoke(
                {
                    "messages": conversation_messages,
                    "next_agent": "supervisor",
                    "visited_agents": [],
                    "mode": mode,
                }
            )
    except Exception as exc:
        console.print(f"[red]Agent Execution Error:[/red] {exc}")
        return conversation_messages

    updated = list(state_out["messages"])
    _render_agent_turn(updated, start_idx=prev_len)
    return updated


@app.command("chat")
def chat_cmd() -> None:
    """Start an interactive multi-turn CLI coaching session with Kinesis."""
    console.print(
        Panel(
            Markdown(
                "# ⚡ Kinesis — Hybrid Training AI Coach (Running + Strength)\n"
                "**Architecture**: LangGraph Supervisor + 3 Sub-Agents (*Analyst*, *Sports Science Researcher*, *Coach & Planner*)\n\n"
                "**Quick Commands**:\n"
                "- `/status` — Check Intervals.icu & LLM connection\n"
                "- `/plan [request]` — Plan one or more workouts (e.g. `/plan 45m easy run tomorrow + strength Friday`)\n"
                "- `/analyze` — Run instant 28-day Intervals.icu readiness & polarization analysis\n"
                "- `/evidence` — View curated peer-reviewed hybrid training evidence (PMIDs & DOIs)\n"
                "- `/clear` — Reset conversation history\n"
                "- `/quit` — Exit"
            ),
            border_style="green",
        )
    )

    graph = build_hybrid_training_graph()
    conversation_messages: list[AnyMessage] = []

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye! Train smart and recover well.[/dim]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/quit", "/exit", "exit", "quit"):
            console.print("[dim]Goodbye! Train smart and recover well.[/dim]")
            break
        if cmd == "/status":
            status_cmd()
            continue
        if cmd == "/analyze":
            analyze_cmd(days=28, ai=False)
            continue
        if cmd == "/evidence":
            evidence_cmd(query="", topic="all")
            continue
        if cmd == "/clear":
            conversation_messages.clear()
            console.print("[dim]Conversation history cleared.[/dim]")
            continue

        if user_input == "/plan" or cmd.startswith("/plan "):
            plan_request = user_input[len("/plan"):].strip()
            if not plan_request:
                try:
                    plan_request = Prompt.ask(
                        "[bold green]Plan request[/bold green] "
                        "[dim](e.g. 45m easy run tomorrow + 60m full body strength Friday)[/dim]"
                    ).strip()
                except (KeyboardInterrupt, EOFError):
                    continue
            if not plan_request:
                continue
            conversation_messages.append(HumanMessage(content=plan_request))
            conversation_messages = _execute_graph_turn(
                graph,
                conversation_messages,
                mode="plan",
                status_text="[bold cyan]Planning workout(s): Analyst → Researcher (if evidence asked) → Planner...[/bold cyan]",
            )
            continue

        conversation_messages.append(HumanMessage(content=user_input))
        conversation_messages = _execute_graph_turn(
            graph,
            conversation_messages,
            mode="auto",
            status_text="[bold cyan]Supervisor coordinating Researcher, Analyst, and Planner sub-agents...[/bold cyan]",
        )


@app.command("tui")
def tui_cmd() -> None:
    """Launch the full-screen interactive Textual TUI (Chat + Sidebar + Tabs)."""
    from hybrid_training_agent.tui import run_tui

    run_tui()


@app.command("mcp")
def mcp_cmd() -> None:
    """Run the Kinesis MCP Server over stdio (for OpenCode / Claude Desktop)."""
    from hybrid_training_agent.mcp_server import run_mcp_server

    run_mcp_server()


@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        tui_cmd()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
