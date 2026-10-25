from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.theme import Theme
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Input,
    Label,
    Markdown,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from hybrid_training_agent.agents.graph import build_hybrid_training_graph
from hybrid_training_agent.cli import _pending_question, extract_message_text
from hybrid_training_agent.clients.intervals_client import (
    IntervalsIcuClient,
    IntervalsIcuError,
)
from hybrid_training_agent.clients.science_client import ScienceLiteratureClient
from hybrid_training_agent.config import Settings, get_settings
from hybrid_training_agent.tools.analytics_tools import analyze_training_and_wellness
from hybrid_training_agent.tools.planner_tools import (
    apply_training_plan_folder,
    delete_planned_event_from_intervals,
    get_athlete_training_plan,
    get_upcoming_intervals_calendar,
    get_workout_library,
    save_workouts_to_library,
    update_calendar_event,
    upload_training_plan_to_intervals,
)

MUTATING_TOOL_MAP = {
    "upload_training_plan_to_intervals": upload_training_plan_to_intervals,
    "delete_planned_event_from_intervals": delete_planned_event_from_intervals,
    "update_calendar_event": update_calendar_event,
    "save_workouts_to_library": save_workouts_to_library,
    "apply_training_plan_folder": apply_training_plan_folder,
}

WELCOME_MARKDOWN = (
    "**Kinesis** — Hybrid Running & Strength Coach\n\n"
    "Ask anything, or use `/plan <workout>`, `/analyze [days]`, `/evidence [query]`, `/status`, `/clear`."
)


@dataclass
class PendingConfirmation:
    """Represents a mutating intervals.icu tool call awaiting explicit user approval."""

    tool_name: str
    tool_args: dict[str, Any]
    action: str
    summary: str
    details: dict[str, Any]


def extract_pending_confirmations(
    messages: list[AnyMessage], start_idx: int = 0
) -> list[PendingConfirmation]:
    """Inspect messages from a turn and extract any unconfirmed mutating tool previews."""
    slice_msgs = messages[start_idx:]
    tool_calls_by_id: dict[str, dict[str, Any]] = {}
    for msg in slice_msgs:
        if isinstance(msg, AIMessage):
            for tc in getattr(msg, "tool_calls", None) or []:
                if isinstance(tc, dict) and tc.get("id"):
                    tool_calls_by_id[str(tc["id"])] = tc

    pending: list[PendingConfirmation] = []
    for msg in slice_msgs:
        if not isinstance(msg, ToolMessage):
            continue
        payload: dict[str, Any] | None = None
        if isinstance(msg.content, dict):
            payload = msg.content
        elif isinstance(msg.content, str):
            text = msg.content.strip()
            if text.startswith("{") and text.endswith("}"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        payload = parsed
                except Exception:
                    payload = None

        if not payload:
            continue

        status = payload.get("status")
        action = str(payload.get("action") or msg.name or "")
        if status == "confirmation_required" and action in MUTATING_TOOL_MAP:
            tc = tool_calls_by_id.get(str(msg.tool_call_id or ""), {})
            raw_args = dict(tc.get("args") or {})
            details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
            if not raw_args:
                raw_args = _reconstruct_tool_args(action, details)
            pending.append(
                PendingConfirmation(
                    tool_name=action,
                    tool_args=raw_args,
                    action=action,
                    summary=str(payload.get("summary") or f"Confirm {action}"),
                    details=details,
                )
            )
        elif status in ("uploaded", "deleted", "updated", "saved", "applied"):
            pending = [p for p in pending if p.action != (msg.name or action)]

    return pending


def _reconstruct_tool_args(action: str, details: dict[str, Any]) -> dict[str, Any]:
    """Fallback reconstruction of tool arguments from the confirmation preview details."""
    if action == "upload_training_plan_to_intervals":
        return {
            "workouts": details.get("workouts", []),
            "upsert": details.get("upsert", True),
        }
    if action == "delete_planned_event_from_intervals":
        event = details.get("event") or {}
        return {"event_id": str(event.get("id") or details.get("event_id") or "")}
    if action == "update_calendar_event":
        changes = details.get("changes") or {}
        updates = {k: v.get("to") for k, v in changes.items() if isinstance(v, dict)}
        return {"event_id": str(details.get("event_id") or ""), "updates": updates}
    if action == "save_workouts_to_library":
        return {"workouts": details.get("workouts", [])}
    if action == "apply_training_plan_folder":
        return {
            "folder_id": details.get("folder_id"),
            "start_date_local": details.get("start_date_local", ""),
        }
    return {}


def execute_pending_confirmation(pending: PendingConfirmation) -> dict[str, Any]:
    """Execute a pending mutating tool call with `confirm=True`."""
    tool_obj = MUTATING_TOOL_MAP.get(pending.tool_name)
    if tool_obj is None:
        return {"status": "failed", "error": f"Unknown tool '{pending.tool_name}'"}
    args = {**pending.tool_args, "confirm": True}
    try:
        res = tool_obj.invoke(args)
        return res if isinstance(res, dict) else {"status": "completed", "result": res}
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "error": str(exc)}


def format_confirmation_result_markdown(
    pending: PendingConfirmation, result: dict[str, Any]
) -> str:
    """Format the result of an approved mutating action as Markdown."""
    status = result.get("status", "unknown")
    if status == "failed" or "error" in result:
        return f"**Failed (`{pending.action}`)**: {result.get('error', 'Unknown error')}"
    if status == "uploaded":
        events = result.get("events") or []
        lines = [
            f"**Uploaded {result.get('uploaded_count', len(events))} Workout(s) to Intervals.icu**",
        ]
        for ev in events[:10]:
            if isinstance(ev, dict):
                lines.append(
                    f"- `{ev.get('start_date_local', '')[:10]}` **{ev.get('name', 'Workout')}** (`{ev.get('type', '')}`)"
                )
        return "\n".join(lines)
    if status == "deleted":
        return f"**Deleted Event** `{result.get('event_id')}` — {pending.summary}"
    if status == "updated":
        return f"**Updated Event** `{result.get('event_id')}` — {pending.summary}"
    if status == "saved":
        return f"**Saved {result.get('saved_count', 0)} Workout(s)** to Library"
    if status == "applied":
        return f"**Applied Plan** `{result.get('folder_id')}` starting `{result.get('start_date_local')}`"
    return f"**Completed (`{pending.action}`)**: {pending.summary}"


def format_status_markdown(
    settings: Settings,
    profile: dict[str, Any] | None = None,
    error: str | None = None,
) -> str:
    """Render minimal system configuration and intervals.icu connection status."""
    try:
        provider = settings.resolve_provider()
    except ValueError:
        provider = "unconfigured"

    model = settings.llm_model or "default"
    lines = [
        "**System**",
        f"LLM: `{provider}` · `{model}`",
    ]
    if profile:
        lines.append(
            f"Athlete: **{profile.get('name', 'Athlete')}** (`{profile.get('icu_resting_hr', '—')} bpm`)"
        )
    elif error:
        lines.append(f"Intervals.icu: {error}")
    elif not settings.has_intervals_credentials:
        lines.append("Intervals.icu: *No API key*")
    return "\n\n".join(lines)


def format_readiness_sidebar_markdown(report: dict[str, Any]) -> str:
    """Render compact readiness and polarization metrics for the sidebar."""
    if not report or "error" in report:
        err = report.get("error", "No data") if isinstance(report, dict) else "No data"
        return f"**Readiness Snapshot**\n\n*{err}*"

    lf = report.get("load_and_form", {})
    rw = report.get("recovery_and_wellness", {})
    rs = report.get("running_summary", {})
    ss = report.get("strength_summary", {})
    audit = report.get("concurrent_interference_audit", {})
    dist = rs.get("intensity_distribution_3_zone", {})
    hrv = rw.get("hrv", {})
    rhr = rw.get("resting_hr", {})
    sleep = rw.get("sleep_and_subjective", {})

    days = report.get("analysis_window_days", 28)
    return (
        f"**Readiness ({days}d)**\n\n"
        f"CTL `{lf.get('ctl_fitness', '—')}` · ATL `{lf.get('atl_fatigue', '—')}` · TSB **`{lf.get('tsb_form', '—')}`**\n"
        f"ACWR `{lf.get('acwr_ratio', '—')}`\n"
        f"HRV `{hrv.get('rolling_7d_mean', '—')}` / `{hrv.get('baseline_mean', '—')}` ms\n"
        f"RHR `{rhr.get('latest_rhr_bpm', '—')}` bpm · Sleep `{sleep.get('rolling_7d_avg_hours', '—')}h`\n"
        f"Run `{rs.get('weekly_avg_distance_km', '—')} km/wk` ({rs.get('sessions_per_week', '—')}/wk)\n"
        f"Z1-2 `{dist.get('zone_1_2_low_pct', 0)}%` · Z3 `{dist.get('zone_3_moderate_grey_zone_pct', 0)}%` · Z4-5 `{dist.get('zone_4_5_high_pct', 0)}%`\n"
        f"Strength `{ss.get('total_sessions', 0)}` ({ss.get('sessions_per_week', 0)}/wk) · Alerts `{audit.get('warnings_count', 0)}`"
    )


def format_analytics_dashboard_markdown(report: dict[str, Any]) -> str:
    """Render quantitative analysis report for the Analytics tab and `/analyze`."""
    if not report or "error" in report:
        err = report.get("error", "Intervals.icu credentials not configured.")
        return f"### Analytics Unavailable\n{err}"

    days = report.get("analysis_window_days", 28)
    lf = report.get("load_and_form", {})
    rw = report.get("recovery_and_wellness", {})
    rs = report.get("running_summary", {})
    ss = report.get("strength_summary", {})
    audit = report.get("concurrent_interference_audit", {})
    dist = rs.get("intensity_distribution_3_zone", {})
    hrv = rw.get("hrv", {})
    rhr = rw.get("resting_hr", {})
    sleep = rw.get("sleep_and_subjective", {})

    md = (
        f"### Load & Form ({days}d)\n"
        f"- **CTL**: `{lf.get('ctl_fitness')}` · **ATL**: `{lf.get('atl_fatigue')}` · **TSB**: `{lf.get('tsb_form')}`\n"
        f"- **ACWR**: `{lf.get('acwr_ratio')}` — *{lf.get('acwr_interpretation')}*\n\n"
        f"### Autonomic Readiness\n"
        f"- **HRV**: `{hrv.get('latest_hrv_rmssd')}` ms (7d `{hrv.get('rolling_7d_mean')}` vs base `{hrv.get('baseline_mean')}`, CV `{hrv.get('rolling_7d_cv_pct')}%`)\n"
        f"- **RHR**: `{rhr.get('latest_rhr_bpm')}` bpm (7d `{rhr.get('rolling_7d_mean_bpm')}`) · **Sleep**: `{sleep.get('rolling_7d_avg_hours')}` h\n\n"
        f"### Running & Polarization\n"
        f"- **Volume**: `{rs.get('weekly_avg_distance_km')} km/wk` across `{rs.get('total_sessions')}` sessions (`{rs.get('sessions_per_week')}/wk`)\n"
        f"- **Zones**: Low `{dist.get('zone_1_2_low_pct')}%` · Mod `{dist.get('zone_3_moderate_grey_zone_pct')}%` · High `{dist.get('zone_4_5_high_pct')}%` (PI `{dist.get('polarization_index')}`)\n\n"
        f"### Strength & Spacing\n"
        f"- **Sessions**: `{ss.get('total_sessions')}` (`{ss.get('sessions_per_week')}/wk`, `{ss.get('total_duration_hours')} h`)\n"
        f"- **Interference Alerts**: `{audit.get('warnings_count', 0)}`"
    )
    if audit.get("warnings"):
        md += "\n" + "\n".join(f"  - {w}" for w in audit["warnings"])
    return md


def format_calendar_sidebar_markdown(calendar_data: dict[str, Any]) -> str:
    """Render upcoming calendar events for the sidebar."""
    if not calendar_data or "error" in calendar_data:
        err = calendar_data.get("error", "No calendar data.") if isinstance(calendar_data, dict) else ""
        return f"**Upcoming**\n\n*{err}*"

    events = calendar_data.get("events") or []
    if not events:
        return "**Upcoming (14d)**\n\n*No planned workouts.*"

    lines = [f"**Upcoming ({len(events)})**\n"]
    for ev in events[:6]:
        date_str = str(ev.get("start_date_local") or "")[5:10]
        sport = ev.get("type") or ev.get("category") or ""
        name = ev.get("name") or "Untitled"
        lines.append(f"`{date_str}` {name} *({sport})*")
    if len(events) > 6:
        lines.append(f"*+{len(events) - 6} more*")
    return "\n".join(lines)


def format_library_markdown(
    library_data: dict[str, Any], plan_data: dict[str, Any] | None = None
) -> str:
    """Render saved workouts and folders from the athlete's intervals.icu library."""
    if not library_data or "error" in library_data:
        err = library_data.get("error", "Library unavailable.") if isinstance(library_data, dict) else ""
        return f"**Library**\n\n*{err}*"

    folders = library_data.get("folders") or []
    workouts = library_data.get("workouts") or []
    lines = [f"### Library ({len(folders)} Folders, {len(workouts)} Workouts)"]
    if plan_data and isinstance(plan_data, dict) and not plan_data.get("error"):
        plan_name = plan_data.get("training_plan_alias") or plan_data.get("name") or plan_data.get("id")
        if plan_name:
            lines.append(f"Active Plan: `{plan_name}`")

    for f in folders[:10]:
        lines.append(f"- **[{f.get('id')}] {f.get('name')}** ({f.get('num_workouts', 0)} workouts)")
    for w in workouts[:15]:
        mins = round(float(w["moving_time"]) / 60) if w.get("moving_time") else "—"
        lines.append(f"- {w.get('name')} (`{w.get('type', 'Run')}`, {mins}m)")
    if not folders and not workouts:
        lines.append("*Empty*")
    return "\n".join(lines)


def format_evidence_markdown(
    papers: list[dict[str, Any]] | None = None,
    curated: dict[str, list[dict[str, Any]]] | None = None,
    query: str = "",
) -> str:
    """Render either live PubMed/Europe PMC results or curated hybrid training evidence."""
    if query.strip():
        if not papers:
            return f"### Search: `{query}`\n*No peer-reviewed papers found.*"
        lines = [f"### Literature Results: `{query}`\n"]
        for idx, p in enumerate(papers, start=1):
            lines.append(
                f"**{idx}. {p.get('title', 'Untitled')}**\n"
                f"{p.get('authors', 'N/A')} ({p.get('year', 'N/A')}) · *{p.get('journal', 'N/A')}* · "
                f"PMID `{p.get('pmid', 'N/A')}` · DOI `{p.get('doi', 'N/A')}`\n\n"
                f"{str(p.get('abstract') or '')[:600]}...\n"
            )
        return "\n".join(lines)

    if not curated:
        return "*No evidence loaded.*"

    lines = ["### Curated Sports Science Evidence\n"]
    for cat, citations in curated.items():
        lines.append(f"#### `{cat}`")
        for c in citations:
            lines.append(
                f"- **{c.get('authors')} ({c.get('year')})** — *{c.get('title')}*\n"
                f"  PMID `{c.get('pmid')}` · {c.get('practical_application')}\n"
            )
    return "\n".join(lines)


class ChatMessageCard(Container):
    """Minimal message block with a left accent line."""

    def __init__(
        self,
        title: str,
        body_markdown: str,
        role_class: str = "agent-planner",
        **kwargs: Any,
    ) -> None:
        super().__init__(classes=f"chat-card {role_class}", **kwargs)
        self._title = title
        self._body_markdown = body_markdown

    def compose(self) -> ComposeResult:
        yield Label(self._title, classes="chat-card-title")
        yield Markdown(self._body_markdown, classes="chat-card-body")


class ConfirmationBanner(Container):
    """Compact inline confirmation bar shown when a mutating tool needs approval."""

    def compose(self) -> ComposeResult:
        with Horizontal(id="confirmation-inner"):
            yield Static("", id="confirmation-summary")
            yield Button("Approve", id="btn-approve-confirm", variant="success")
            yield Button("Reject", id="btn-reject-confirm", variant="error")


KINESIS_MONO_RED_THEME = Theme(
    name="kinesis-mono-red",
    primary="#f28b82",
    secondary="#c96f69",
    accent="#ffb4ab",
    foreground="#f2dede",
    background="#0d0809",
    surface="#140c0e",
    panel="#1c1113",
    warning="#f28b82",
    error="#e57373",
    success="#f28b82",
    dark=True,
)


class KinesisTUI(App[None]):
    """Minimalistic monochromatic light-red full-screen Textual TUI for Kinesis."""

    TITLE = "kinesis"

    CSS = """
    Screen {
        background: #0d0809;
        color: #f2dede;
    }

    #top-bar {
        height: 3;
        padding: 0 3;
        align: left middle;
        background: #140c0e;
        border-bottom: solid #2b181b;
        color: #9e6b70;
    }

    #top-brand {
        width: auto;
        height: 1;
        content-align: left middle;
        color: #ffb4ab;
        text-style: bold;
    }

    #top-telemetry {
        width: 1fr;
        height: 1;
        content-align: right middle;
        color: #b87c82;
    }

    #main-tabs {
        height: 1fr;
        background: #0d0809;
    }

    Tabs {
        height: 3;
        padding: 0 2;
        background: #110a0c;
        border-bottom: solid #241417;
    }

    Tab {
        height: 3;
        padding: 0 3;
        content-align: center middle;
        color: #85585c;
        background: transparent;
    }

    Tab:hover {
        color: #e6c3c3;
    }

    Tab.-active {
        color: #ffb4ab;
        text-style: bold;
    }

    Underline > .underline--bar {
        color: #f28b82;
        background: #2b181b;
    }

    TabPane {
        padding: 0;
        background: #0d0809;
    }

    #chat-workspace {
        height: 1fr;
    }

    #chat-main {
        width: 1fr;
        height: 1fr;
        padding: 0 3;
    }

    #chat-scroll {
        height: 1fr;
        padding: 1 0;
        border: none;
        background: #0d0809;
    }

    .chat-card {
        height: auto;
        margin: 0 0 1 0;
        padding: 0 0 0 2;
        border: none;
        border-left: thick #b8645e;
        background: transparent;
    }

    .chat-card-title {
        text-style: bold;
        color: #b8645e;
        margin-bottom: 0;
    }

    .role-user {
        border-left: thick #ffb4ab;
    }
    .role-user .chat-card-title {
        color: #ffb4ab;
    }

    .agent-analyst {
        border-left: thick #e57373;
    }
    .agent-analyst .chat-card-title {
        color: #e57373;
    }

    .agent-researcher {
        border-left: thick #c96f69;
    }
    .agent-researcher .chat-card-title {
        color: #c96f69;
    }

    .agent-planner {
        border-left: thick #f28b82;
    }
    .agent-planner .chat-card-title {
        color: #f28b82;
    }

    .role-system {
        border-left: thick #9e6b70;
    }
    .role-system .chat-card-title {
        color: #9e6b70;
    }

    .tool-strip {
        height: auto;
        color: #85585c;
        padding: 0 0 0 2;
        margin: 0 0 1 0;
    }

    #agent-activity-bar {
        height: auto;
        min-height: 1;
        padding: 0 1;
        margin: 0 0 1 0;
        color: #c96f69;
    }

    #confirmation-bar {
        display: none;
        height: auto;
        margin: 0 0 1 0;
        padding: 1 2;
        border-left: thick #ffb4ab;
        background: #1a1012;
    }

    #confirmation-bar.visible {
        display: block;
    }

    #confirmation-inner {
        height: auto;
        align: left middle;
    }

    #confirmation-summary {
        width: 1fr;
        color: #ffb4ab;
        padding: 0 2 0 0;
    }

    #confirmation-inner Button {
        min-width: 12;
        height: 3;
        margin-left: 1;
    }

    Button, Button.-style-default {
        min-width: 11;
        height: 3;
        background: #1c1113;
        color: #c98a8e;
        border: none;
        border-top: tall #2b181b;
        border-bottom: tall #0d0809;
    }

    Button:hover, Button:focus, Button.-style-default:hover, Button.-style-default:focus {
        background: #2b181b;
        color: #ffb4ab;
        border-top: tall #f28b82;
        border-bottom: tall #1c1113;
    }

    Button.-primary, Button.-success, Button.-style-default.-primary, Button.-style-default.-success, Button.active-window {
        background: #331b1f;
        color: #ffb4ab;
        border-top: tall #f28b82;
        border-bottom: tall #241215;
        text-style: bold;
    }

    Button.-error, Button.-style-default.-error {
        background: #1c1113;
        color: #c96f69;
        border-top: tall #5c3236;
        border-bottom: tall #0d0809;
    }

    #chat-input-bar {
        height: 5;
        padding: 1 0;
        align: left middle;
    }

    #mode-select {
        display: none;
    }

    #chat-input, #evidence-query-input {
        width: 1fr;
        height: 3;
        padding: 0 2;
        border: round #4a292e;
        background: #140c0e;
        color: #f2dede;
    }

    #chat-input:focus, #evidence-query-input:focus {
        border: round #f28b82;
        background: #180e11;
    }

    #btn-send {
        display: none;
    }

    #sidebar {
        width: 36;
        height: 1fr;
        border-left: solid #2b181b;
        padding: 1 2;
        background: #110a0c;
    }

    #sidebar.hidden {
        display: none;
    }

    .sidebar-card {
        height: auto;
        margin: 0 0 1 0;
        padding: 0 0 1 0;
        border: none;
        border-bottom: solid #2b181b;
        background: transparent;
    }

    #sidebar-pipeline {
        color: #9e6b70;
    }

    .tab-toolbar {
        height: 5;
        padding: 1 3;
        align: left middle;
        background: #140c0e;
        border-bottom: solid #241417;
    }

    .tab-toolbar Button {
        margin-right: 1;
        min-width: 10;
    }

    .tab-toolbar Button.day-btn {
        min-width: 7;
    }

    #analytics-status {
        width: 1fr;
        height: 3;
        content-align: right middle;
        color: #c96f69;
        padding: 0 1;
    }

    #analytics-body, #calendar-body {
        height: 1fr;
        padding: 1 3;
    }

    #analytics-report-scroll {
        width: 3fr;
        height: 1fr;
        padding: 0 2 0 0;
    }

    #analytics-ai-markdown {
        display: none;
        height: auto;
        margin: 0 0 1 0;
        padding: 1 2;
        border-left: thick #ffb4ab;
        background: #160d0f;
    }

    #analytics-ai-markdown.visible {
        display: block;
    }

    #analytics-activities-pane {
        width: 2fr;
        height: 1fr;
        border-left: solid #2b181b;
        padding: 0 0 0 2;
    }

    #activities-table {
        height: 1fr;
        margin-bottom: 1;
    }

    #activity-detail-scroll {
        height: 9;
        border-top: solid #2b181b;
        padding: 1 0;
    }

    #calendar-events-pane {
        width: 3fr;
        height: 1fr;
        padding: 0 2 0 0;
    }

    #calendar-table {
        height: 1fr;
        margin-bottom: 1;
    }

    DataTable {
        background: #0d0809;
        color: #e6c3c3;
    }

    DataTable > .datatable--header {
        background: #1a1012;
        color: #ffb4ab;
        text-style: bold;
    }

    DataTable > .datatable--cursor {
        background: #331c20;
        color: #ffb4ab;
    }

    #calendar-detail-scroll {
        height: 10;
        border-top: solid #2b181b;
        padding: 1 0;
    }

    #library-pane {
        width: 2fr;
        height: 1fr;
        border-left: solid #2b181b;
        padding: 0 0 0 2;
    }

    .pane-title {
        text-style: bold;
        color: #c96f69;
        margin-bottom: 1;
    }

    #evidence-topic-select {
        width: 28;
        margin-right: 1;
    }

    #evidence-scroll {
        height: 1fr;
        padding: 1 3;
    }

    #bottom-bar {
        height: 3;
        padding: 0 3;
        align: left middle;
        background: #140c0e;
        border-top: solid #2b181b;
        color: #9e6b70;
    }

    #bottom-hints {
        width: 1fr;
        height: 1;
        content-align: left middle;
        color: #9e6b70;
    }

    #bottom-keys {
        width: auto;
        height: 1;
        content-align: right middle;
        color: #b87c82;
    }
    """

    BINDINGS = [
        Binding("ctrl+b", "toggle_sidebar", "Sidebar", show=True),
        Binding("ctrl+r", "refresh_data", "Refresh", show=True),
        Binding("ctrl+l", "clear_chat", "Clear", show=True),
        Binding("f1", "tab_chat", "Chat", show=False),
        Binding("f2", "tab_analytics", "Analytics", show=False),
        Binding("f3", "tab_calendar", "Calendar", show=False),
        Binding("f4", "tab_evidence", "Evidence", show=False),
        Binding("ctrl+p", "quick_plan", "Plan", show=False),
        Binding("ctrl+q", "quit", "Quit", show=True),
    ]

    def __init__(
        self,
        graph: Any = None,
        auto_refresh_on_mount: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.register_theme(KINESIS_MONO_RED_THEME)
        self.theme = "kinesis-mono-red"
        self._graph = graph
        self._auto_refresh_on_mount = auto_refresh_on_mount
        self.conversation_messages: list[AnyMessage] = []
        self.pending_confirmations: list[PendingConfirmation] = []
        self.analytics_days: int = 28
        self._activities_cache: list[dict[str, Any]] = []
        self._calendar_events_cache: list[dict[str, Any]] = []
        self._is_busy: bool = False

    def _get_graph(self) -> Any:
        if self._graph is None:
            self._graph = build_hybrid_training_graph()
        return self._graph

    def compose(self) -> ComposeResult:
        with Horizontal(id="top-bar"):
            yield Static("⚡ kinesis", id="top-brand")
            yield Static("ctrl+b sidebar · /plan · /analyze · /evidence", id="top-telemetry")

        with TabbedContent(id="main-tabs"):
            with TabPane("Chat", id="tab-chat"):
                with Horizontal(id="chat-workspace"):
                    with Vertical(id="chat-main"):
                        with VerticalScroll(id="chat-scroll"):
                            yield ChatMessageCard(
                                title="kinesis",
                                body_markdown=WELCOME_MARKDOWN,
                                role_class="agent-planner",
                                id="welcome-card",
                            )
                        yield Static("", id="agent-activity-bar")
                        yield ConfirmationBanner(id="confirmation-bar")
                        with Horizontal(id="chat-input-bar"):
                            yield Select(
                                options=[
                                    ("auto", "auto"),
                                    ("plan", "plan"),
                                    ("analyze", "analyze"),
                                    ("research", "research"),
                                ],
                                value="auto",
                                allow_blank=False,
                                id="mode-select",
                            )
                            yield Input(
                                placeholder="Ask anything or type /plan, /analyze, /evidence, /clear...",
                                id="chat-input",
                            )
                            yield Button("Send", id="btn-send", variant="primary")
                    with VerticalScroll(id="sidebar"):
                        yield Static(
                            "Route: auto",
                            id="sidebar-pipeline",
                            classes="sidebar-card",
                        )
                        yield Markdown(
                            "**Readiness**\n\n*Loading...*",
                            id="sidebar-readiness",
                            classes="sidebar-card",
                        )
                        yield Markdown(
                            "**Upcoming**\n\n*Loading...*",
                            id="sidebar-calendar",
                            classes="sidebar-card",
                        )
                        yield Markdown(
                            "**System**\n\n*Checking...*",
                            id="sidebar-status",
                            classes="sidebar-card",
                        )

            with TabPane("Analytics", id="tab-analytics"):
                with Horizontal(id="analytics-toolbar", classes="tab-toolbar"):
                    yield Button("7d", id="btn-days-7", classes="day-btn")
                    yield Button("14d", id="btn-days-14", classes="day-btn")
                    yield Button(
                        "28d",
                        id="btn-days-28",
                        variant="primary",
                        classes="day-btn active-window",
                    )
                    yield Button("60d", id="btn-days-60", classes="day-btn")
                    yield Button("90d", id="btn-days-90", classes="day-btn")
                    yield Button("Refresh", id="btn-refresh-analytics")
                    yield Button("AI Commentary", id="btn-ai-analytics", variant="success")
                    yield Static("Window: 28d", id="analytics-status")
                with Horizontal(id="analytics-body"):
                    with VerticalScroll(id="analytics-report-scroll"):
                        yield Markdown(
                            "",
                            id="analytics-ai-markdown",
                        )
                        yield Markdown(
                            "*Select a window or click Refresh.*",
                            id="analytics-markdown",
                        )
                    with Vertical(id="analytics-activities-pane"):
                        yield Label("Recent Activities", classes="pane-title")
                        yield DataTable(id="activities-table", cursor_type="row")
                        with VerticalScroll(id="activity-detail-scroll"):
                            yield Markdown(
                                "*Select an activity row to inspect metrics.*",
                                id="activity-detail",
                            )

            with TabPane("Calendar", id="tab-calendar"):
                with Horizontal(id="calendar-toolbar", classes="tab-toolbar"):
                    yield Button("Refresh", id="btn-refresh-calendar", variant="primary")
                    yield Button("+ Plan Workout", id="btn-goto-plan", variant="success")
                with Horizontal(id="calendar-body"):
                    with Vertical(id="calendar-events-pane"):
                        yield Label("Upcoming (14d)", classes="pane-title")
                        yield DataTable(id="calendar-table", cursor_type="row")
                        with VerticalScroll(id="calendar-detail-scroll"):
                            yield Markdown(
                                "*Select an event to view workout steps.*",
                                id="calendar-event-detail",
                            )
                    with VerticalScroll(id="library-pane"):
                        yield Markdown(
                            "**Library**\n\n*Loading...*",
                            id="library-markdown",
                        )

            with TabPane("Evidence", id="tab-evidence"):
                with Horizontal(id="evidence-toolbar", classes="tab-toolbar"):
                    yield Select(
                        options=[
                            ("All Topics", "all"),
                            ("Interference", "concurrent_interference"),
                            ("Running Economy", "running_economy_and_strength"),
                            ("Polarization", "intensity_distribution_and_polarization"),
                            ("HRV & Load", "hrv_and_load_monitoring"),
                        ],
                        value="all",
                        allow_blank=False,
                        id="evidence-topic-select",
                    )
                    yield Input(
                        placeholder="Search PubMed / Europe PMC...",
                        id="evidence-query-input",
                    )
                    yield Button("Search", id="btn-search-evidence", variant="primary")
                    yield Button("Curated", id="btn-curated-evidence")
                with VerticalScroll(id="evidence-scroll"):
                    yield Markdown(
                        "*Loading...*",
                        id="evidence-markdown",
                    )

        with Horizontal(id="bottom-bar"):
            yield Static("/plan · /analyze · /evidence · /status · /clear", id="bottom-hints")
            yield Static("^b sidebar   ^r refresh   ^l clear   ^q quit", id="bottom-keys")

    async def on_mount(self) -> None:
        activities_table = self.query_one("#activities-table", DataTable)
        activities_table.add_columns("Date", "Sport", "Name", "km", "Min", "Load")

        calendar_table = self.query_one("#calendar-table", DataTable)
        calendar_table.add_columns("Date", "Type", "Category", "Name", "Min", "Load", "ID")

        self.query_one("#chat-input", Input).focus()
        self._load_curated_evidence("all")

        if self._auto_refresh_on_mount:
            self.run_worker(self._refresh_all_data_async(), exclusive=False)

    # ------------------------------------------------------------------
    # Keybinding Actions
    # ------------------------------------------------------------------
    def action_toggle_sidebar(self) -> None:
        sidebar = self.query_one("#sidebar", VerticalScroll)
        sidebar.toggle_class("hidden")

    def action_refresh_data(self) -> None:
        self.run_worker(self._refresh_all_data_async(), exclusive=False)

    def action_quick_plan(self) -> None:
        self.query_one("#main-tabs", TabbedContent).active = "tab-chat"
        self.query_one("#mode-select", Select).value = "plan"
        chat_input = self.query_one("#chat-input", Input)
        if not chat_input.value.startswith("/plan"):
            chat_input.value = "/plan "
            chat_input.cursor_position = len(chat_input.value)
        chat_input.focus()

    async def action_clear_chat(self) -> None:
        await self._clear_chat_history()

    def action_tab_chat(self) -> None:
        self.query_one("#main-tabs", TabbedContent).active = "tab-chat"
        self.query_one("#chat-input", Input).focus()

    def action_tab_analytics(self) -> None:
        self.query_one("#main-tabs", TabbedContent).active = "tab-analytics"

    def action_tab_calendar(self) -> None:
        self.query_one("#main-tabs", TabbedContent).active = "tab-calendar"

    def action_tab_evidence(self) -> None:
        self.query_one("#main-tabs", TabbedContent).active = "tab-evidence"

    # ------------------------------------------------------------------
    # Data Loading & Background Refresh
    # ------------------------------------------------------------------
    def _load_curated_evidence(self, topic: str = "all") -> None:
        curated = ScienceLiteratureClient.get_curated_evidence(topic)
        md = format_evidence_markdown(curated=curated)
        self.query_one("#evidence-markdown", Markdown).update(md)

    async def _refresh_all_data_async(self) -> None:
        await asyncio.gather(
            self._refresh_status_async(),
            self._refresh_analytics_async(self.analytics_days),
            self._refresh_calendar_and_library_async(),
        )

    async def _refresh_status_async(self) -> str:
        settings = get_settings()
        profile: dict[str, Any] | None = None
        err: str | None = None
        if settings.has_intervals_credentials:
            client = IntervalsIcuClient(settings=settings)
            try:
                profile = await asyncio.to_thread(client.get_athlete_profile)
            except IntervalsIcuError as exc:
                err = str(exc)
        md = format_status_markdown(settings, profile=profile, error=err)
        await self.query_one("#sidebar-status", Markdown).update(md)
        return md

    def _set_active_days_button(self, days: int) -> None:
        self.analytics_days = days
        for w_days in (7, 14, 28, 60, 90):
            btn = self.query_one(f"#btn-days-{w_days}", Button)
            if w_days == days:
                btn.variant = "primary"
                btn.add_class("active-window")
            else:
                btn.variant = "default"
                btn.remove_class("active-window")

    def _format_activity_detail_markdown(self, row: dict[str, Any]) -> str:
        name = row.get("name") or "Workout"
        act_type = row.get("type") or "Activity"
        date_str = row.get("date") or "—"
        lines = [f"**{name}** (`{date_str}` · `{act_type}`)"]
        stats: list[str] = []
        if row.get("distance_km") is not None:
            stats.append(f"Distance: `{row['distance_km']} km`")
        if row.get("duration_min") is not None:
            stats.append(f"Duration: `{row['duration_min']} min`")
        if row.get("avg_pace"):
            stats.append(f"Pace: `{row['avg_pace']}`")
        if row.get("gap_pace"):
            stats.append(f"GAP: `{row['gap_pace']}`")
        if row.get("avg_hr"):
            hr_str = f"`{row['avg_hr']} bpm`"
            if row.get("max_hr"):
                hr_str += f" (max `{row['max_hr']}`)"
            stats.append(f"HR: {hr_str}")
        if row.get("load") is not None:
            stats.append(f"Load: `{row['load']}`")
        if row.get("rpe") is not None:
            stats.append(f"RPE: `{row['rpe']}`")
        if row.get("decoupling_pct") is not None:
            stats.append(f"Decoupling: `{row['decoupling_pct']}%`")
        if stats:
            lines.append(" · ".join(stats))
        return "\n\n".join(lines)

    async def _refresh_analytics_async(self, days: int) -> dict[str, Any]:
        self._set_active_days_button(days)
        status_widget = self.query_one("#analytics-status", Static)
        status_widget.update(f"Loading {days}d...")

        settings = get_settings()
        if not settings.has_intervals_credentials:
            report: dict[str, Any] = {
                "error": "Set INTERVALS_ICU_API_KEY in .env."
            }
        else:
            report = await asyncio.to_thread(
                analyze_training_and_wellness.invoke, {"days_back": days}
            )

        await self.query_one("#sidebar-readiness", Markdown).update(
            format_readiness_sidebar_markdown(report)
        )
        await self.query_one("#analytics-markdown", Markdown).update(
            format_analytics_dashboard_markdown(report)
        )

        if "load_and_form" in report:
            lf = report["load_and_form"]
            hrv = report.get("recovery_and_wellness", {}).get("hrv", {}).get("rolling_7d_mean")
            hrv_part = f" · HRV {hrv}ms" if hrv else ""
            self.query_one("#top-telemetry", Static).update(
                f"CTL {lf.get('ctl_fitness', '—')} · ATL {lf.get('atl_fatigue', '—')} · TSB {lf.get('tsb_form', '—')}{hrv_part}"
            )

        # Build enriched lookup from recent_runs & strength_sessions
        detailed_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for r in report.get("running_summary", {}).get("recent_runs") or []:
            detailed_by_key[(str(r.get("date") or ""), str(r.get("name") or ""))] = {
                **r,
                "type": "Run",
            }
        for s in report.get("strength_summary", {}).get("strength_sessions") or []:
            detailed_by_key[(str(s.get("date") or ""), str(s.get("name") or ""))] = {
                **s,
                "type": "WeightTraining",
            }

        raw_rows = list(reversed(report.get("activity_log") or []))
        enriched_rows: list[dict[str, Any]] = []
        table = self.query_one("#activities-table", DataTable)
        table.clear()
        for row in raw_rows:
            key = (str(row.get("date") or ""), str(row.get("name") or ""))
            merged = {**row, **detailed_by_key.get(key, {})}
            enriched_rows.append(merged)
            table.add_row(
                str(merged.get("date") or "")[5:],
                str(merged.get("type") or ""),
                str(merged.get("name") or "")[:30],
                str(merged.get("distance_km") if merged.get("distance_km") is not None else "—"),
                str(merged.get("duration_min") if merged.get("duration_min") is not None else "—"),
                str(merged.get("load") if merged.get("load") is not None else "—"),
            )

        self._activities_cache = enriched_rows
        if enriched_rows:
            await self.query_one("#activity-detail", Markdown).update(
                self._format_activity_detail_markdown(enriched_rows[0])
            )
            status_widget.update(f"Window: {days}d · {len(enriched_rows)} activities")
        else:
            await self.query_one("#activity-detail", Markdown).update(
                "*No completed activities in this window.*"
            )
            status_widget.update(f"Window: {days}d")

        return report

    async def _refresh_calendar_and_library_async(self) -> None:
        settings = get_settings()
        if not settings.has_intervals_credentials:
            cal_data: dict[str, Any] = {"error": "No API key."}
            lib_data: dict[str, Any] = {"error": "No API key."}
            plan_data: dict[str, Any] = {}
        else:
            cal_data, lib_data, plan_data = await asyncio.gather(
                asyncio.to_thread(get_upcoming_intervals_calendar.invoke, {"days_ahead": 14}),
                asyncio.to_thread(get_workout_library.invoke, {}),
                asyncio.to_thread(get_athlete_training_plan.invoke, {}),
            )

        await self.query_one("#sidebar-calendar", Markdown).update(
            format_calendar_sidebar_markdown(cal_data)
        )
        await self.query_one("#library-markdown", Markdown).update(
            format_library_markdown(lib_data, plan_data)
        )

        self._calendar_events_cache = list(cal_data.get("events") or [])
        cal_table = self.query_one("#calendar-table", DataTable)
        cal_table.clear()
        for ev in self._calendar_events_cache:
            mins = (
                round(float(ev["moving_time"]) / 60)
                if ev.get("moving_time")
                else "—"
            )
            cal_table.add_row(
                str(ev.get("start_date_local") or "")[:10],
                str(ev.get("type") or "—"),
                str(ev.get("category") or "—"),
                str(ev.get("name") or "Untitled")[:36],
                str(mins),
                str(ev.get("icu_training_load") or "—"),
                str(ev.get("id") or ""),
            )

    # ------------------------------------------------------------------
    # Chat Helpers & Message Rendering
    # ------------------------------------------------------------------
    async def _append_chat_card(
        self, title: str, markdown_text: str, role_class: str = "agent-planner"
    ) -> None:
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        card = ChatMessageCard(title=title, body_markdown=markdown_text, role_class=role_class)
        await scroll.mount(card)
        scroll.scroll_end(animate=False)

    async def _append_tool_strip(self, text: str) -> None:
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        strip = Static(text, classes="tool-strip")
        await scroll.mount(strip)
        scroll.scroll_end(animate=False)

    def _set_activity_status(self, text: str) -> None:
        self.query_one("#agent-activity-bar", Static).update(text)

    def _update_pipeline_sidebar(
        self,
        pipeline: list[str] | None = None,
        visited: list[str] | None = None,
        router: str = "",
        mode: str = "auto",
    ) -> None:
        pipe_str = " → ".join(pipeline) if pipeline else "auto"
        self.query_one("#sidebar-pipeline", Static).update(f"Route: {pipe_str}")

    def _update_confirmation_banner(self) -> None:
        banner = self.query_one("#confirmation-bar", ConfirmationBanner)
        summary_widget = self.query_one("#confirmation-summary", Static)
        if self.pending_confirmations:
            summaries = " | ".join(p.summary for p in self.pending_confirmations)
            summary_widget.update(f"Confirm: {summaries}")
            banner.add_class("visible")
        else:
            summary_widget.update("")
            banner.remove_class("visible")

    async def _render_new_messages(
        self, messages: list[AnyMessage], start_idx: int
    ) -> None:
        for msg in messages[start_idx:]:
            if isinstance(msg, ToolMessage):
                await self._append_tool_strip(f"· {msg.name}")
            elif isinstance(msg, AIMessage):
                text = extract_message_text(msg.content).strip()
                if not text:
                    continue
                if getattr(msg, "tool_calls", None):
                    continue

                agent_label = getattr(msg, "name", None) or "kinesis"
                role_class = "agent-planner"
                short_title = "planner"
                if "Researcher" in agent_label:
                    role_class = "agent-researcher"
                    short_title = "researcher"
                elif "Analyst" in agent_label:
                    role_class = "agent-analyst"
                    short_title = "analyst"

                await self._append_chat_card(
                    title=short_title,
                    markdown_text=text,
                    role_class=role_class,
                )

    async def _clear_chat_history(self) -> None:
        self.conversation_messages.clear()
        self.pending_confirmations.clear()
        self._update_confirmation_banner()
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        await scroll.remove_children()
        await scroll.mount(
            ChatMessageCard(
                title="kinesis",
                body_markdown=WELCOME_MARKDOWN,
                role_class="agent-planner",
            )
        )
        self._set_activity_status("")

    # ------------------------------------------------------------------
    # Graph Turn Execution & Slash Commands
    # ------------------------------------------------------------------
    async def submit_user_input(self, raw_input: str) -> None:
        """Process a user chat message or slash command."""
        user_input = raw_input.strip()
        if not user_input or self._is_busy:
            return

        cmd = user_input.lower()
        if cmd in ("/quit", "/exit", "quit", "exit"):
            self.exit()
            return

        if cmd == "/clear":
            await self._clear_chat_history()
            return

        if cmd == "/status":
            await self._append_chat_card("you", "`/status`", role_class="role-user")
            md = await self._refresh_status_async()
            await self._append_chat_card("system", md, role_class="role-system")
            return

        if cmd == "/analyze" or cmd.startswith("/analyze "):
            arg = user_input[len("/analyze") :].strip()
            days = int(arg) if arg.isdigit() else 28
            await self._append_chat_card("you", f"`/analyze {days}`", role_class="role-user")
            self._set_activity_status(f"Analyzing {days}d...")
            report = await self._refresh_analytics_async(days)
            await self._append_chat_card(
                "analyst",
                format_analytics_dashboard_markdown(report),
                role_class="agent-analyst",
            )
            self._set_activity_status("")
            return

        if cmd == "/evidence" or cmd.startswith("/evidence "):
            query = user_input[len("/evidence") :].strip()
            await self._append_chat_card(
                "you",
                f"`/evidence {query}`".strip(),
                role_class="role-user",
            )
            md = await self._run_evidence_lookup_async(query=query, topic="all")
            await self._append_chat_card(
                "researcher",
                md,
                role_class="agent-researcher",
            )
            return

        selected_mode = str(self.query_one("#mode-select", Select).value or "auto")
        prompt_text = user_input
        if user_input == "/plan" or cmd.startswith("/plan "):
            plan_req = user_input[len("/plan") :].strip()
            if not plan_req:
                self.action_quick_plan()
                self._set_activity_status("Enter workout details after /plan...")
                return
            prompt_text = plan_req
            selected_mode = "plan"

        await self._execute_agent_turn_async(prompt_text, mode=selected_mode)

    async def _execute_agent_turn_async(self, prompt_text: str, mode: str = "auto") -> None:
        self._is_busy = True
        try:
            await self._append_chat_card(
                title="you",
                markdown_text=prompt_text,
                role_class="role-user",
            )
            self.conversation_messages.append(HumanMessage(content=prompt_text))
            prev_len = len(self.conversation_messages)
            self._set_activity_status("Thinking...")

            graph = self._get_graph()
            initial_state = {
                "messages": list(self.conversation_messages),
                "next_agent": "supervisor",
                "visited_agents": [],
                "mode": mode,
            }

            state_out = await asyncio.to_thread(graph.invoke, initial_state)
            updated_messages = list(state_out.get("messages") or self.conversation_messages)
            visited = list(state_out.get("visited_agents") or [])
            pipeline = list(state_out.get("pipeline") or visited)
            router = str(state_out.get("router") or "heuristic")

            self.conversation_messages = updated_messages
            self._update_pipeline_sidebar(
                pipeline=pipeline,
                visited=visited,
                router=router,
                mode=mode,
            )
            await self._render_new_messages(self.conversation_messages, start_idx=prev_len)

            self.pending_confirmations = extract_pending_confirmations(
                self.conversation_messages, start_idx=prev_len
            )
            self._update_confirmation_banner()

            if self.pending_confirmations:
                self._set_activity_status("Awaiting confirmation.")
            elif _pending_question(self.conversation_messages):
                self._set_activity_status("Awaiting your reply.")
            else:
                self._set_activity_status("")
        except Exception as exc:  # noqa: BLE001
            await self._append_chat_card(
                title="error",
                markdown_text=f"`{exc}`",
                role_class="role-system",
            )
            self._set_activity_status("")
        finally:
            self._is_busy = False

    async def approve_pending_confirmations(self) -> None:
        """Approve and execute all currently pending intervals.icu confirmations."""
        if not self.pending_confirmations or self._is_busy:
            return
        self._is_busy = True
        to_run = list(self.pending_confirmations)
        self.pending_confirmations.clear()
        self._update_confirmation_banner()
        self._set_activity_status("Syncing with intervals.icu...")

        try:
            self.conversation_messages.append(
                HumanMessage(content="Approved. Proceed with confirm=true.")
            )
            for item in to_run:
                result = await asyncio.to_thread(execute_pending_confirmation, item)
                md = format_confirmation_result_markdown(item, result)
                self.conversation_messages.append(
                    AIMessage(content=md, name="Hybrid_Coach_Planner")
                )
                await self._append_chat_card(
                    title="planner",
                    markdown_text=md,
                    role_class="agent-planner",
                )
            await self._refresh_calendar_and_library_async()
            self._set_activity_status("")
        finally:
            self._is_busy = False

    async def reject_pending_confirmations(self) -> None:
        """Reject and discard currently pending intervals.icu confirmations."""
        if not self.pending_confirmations:
            return
        summaries = ", ".join(p.action for p in self.pending_confirmations)
        self.pending_confirmations.clear()
        self._update_confirmation_banner()
        cancel_msg = f"Cancelled `{summaries}`."
        self.conversation_messages.append(HumanMessage(content="Reject proposed calendar action."))
        self.conversation_messages.append(
            AIMessage(content=cancel_msg, name="Hybrid_Coach_Planner")
        )
        await self._append_chat_card(
            title="system",
            markdown_text=cancel_msg,
            role_class="role-system",
        )
        self._set_activity_status("")

    async def _run_evidence_lookup_async(
        self, query: str = "", topic: str = "all"
    ) -> str:
        client = ScienceLiteratureClient()
        if query.strip():
            self._set_activity_status(f"Searching PubMed for '{query}'...")
            papers = await asyncio.to_thread(client.search_pubmed, query, 5)
            if not papers:
                papers = await asyncio.to_thread(client.search_europe_pmc, query, 5)
            md = format_evidence_markdown(papers=papers, query=query)
            self._set_activity_status("")
        else:
            curated = client.get_curated_evidence(topic)
            md = format_evidence_markdown(curated=curated)
        await self.query_one("#evidence-markdown", Markdown).update(md)
        return md

    # ------------------------------------------------------------------
    # UI Event Handlers
    # ------------------------------------------------------------------
    @on(Input.Submitted, "#chat-input")
    async def _on_chat_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value
        event.input.value = ""
        await self.submit_user_input(text)

    @on(Button.Pressed, "#btn-send")
    async def _on_send_pressed(self) -> None:
        chat_input = self.query_one("#chat-input", Input)
        text = chat_input.value
        chat_input.value = ""
        await self.submit_user_input(text)

    @on(Button.Pressed, "#btn-approve-confirm")
    async def _on_approve_pressed(self) -> None:
        await self.approve_pending_confirmations()

    @on(Button.Pressed, "#btn-reject-confirm")
    async def _on_reject_pressed(self) -> None:
        await self.reject_pending_confirmations()

    def _trigger_analytics_window(self, days: int) -> None:
        self._set_active_days_button(days)
        self.query_one("#analytics-status", Static).update(f"Loading {days}d...")
        self.run_worker(
            self._refresh_analytics_async(days),
            group="analytics",
            exclusive=True,
        )

    async def _run_ai_analytics_async(self) -> None:
        days = self.analytics_days
        ai_md = self.query_one("#analytics-ai-markdown", Markdown)
        ai_btn = self.query_one("#btn-ai-analytics", Button)
        status_widget = self.query_one("#analytics-status", Static)

        ai_btn.label = "Analyzing..."
        ai_btn.disabled = True
        ai_md.add_class("visible")
        await ai_md.update(
            f"### AI Coach Commentary ({days}d)\n*Analyzing your last {days} days of load, HRV readiness, polarization, and strength spacing...*"
        )
        status_widget.update(f"Running AI Analyst ({days}d)...")

        prompt = (
            f"Analyze my past {days} days of training and wellness data from intervals.icu. "
            "Evaluate my CTL/ATL/TSB, ACWR, HRV & RHR recovery trends, running zone polarization, "
            "and strength training frequency/spacing."
        )
        try:
            await self._execute_agent_turn_async(prompt, mode="analyze")
            latest_ai_text = ""
            for msg in reversed(self.conversation_messages):
                if isinstance(msg, AIMessage):
                    text = extract_message_text(msg.content).strip()
                    if text and not getattr(msg, "tool_calls", None):
                        latest_ai_text = text
                        break
            if latest_ai_text:
                await ai_md.update(f"### AI Coach Commentary ({days}d)\n\n{latest_ai_text}")
                status_widget.update(f"AI Commentary ready ({days}d)")
            else:
                await ai_md.update(
                    f"### AI Coach Commentary ({days}d)\n*No commentary returned.*"
                )
                status_widget.update(f"Window: {days}d")
        except Exception as exc:  # noqa: BLE001
            await ai_md.update(f"### AI Coach Commentary Error\n`{exc}`")
            status_widget.update("AI Commentary failed")
        finally:
            ai_btn.label = "AI Commentary"
            ai_btn.disabled = False

    @on(Button.Pressed, "#btn-days-7")
    def _on_days_7(self) -> None:
        self._trigger_analytics_window(7)

    @on(Button.Pressed, "#btn-days-14")
    def _on_days_14(self) -> None:
        self._trigger_analytics_window(14)

    @on(Button.Pressed, "#btn-days-28")
    def _on_days_28(self) -> None:
        self._trigger_analytics_window(28)

    @on(Button.Pressed, "#btn-days-60")
    def _on_days_60(self) -> None:
        self._trigger_analytics_window(60)

    @on(Button.Pressed, "#btn-days-90")
    def _on_days_90(self) -> None:
        self._trigger_analytics_window(90)

    @on(Button.Pressed, "#btn-refresh-analytics")
    def _on_refresh_analytics(self) -> None:
        self._trigger_analytics_window(self.analytics_days)

    @on(Button.Pressed, "#btn-ai-analytics")
    def _on_ai_analytics(self) -> None:
        self.run_worker(
            self._run_ai_analytics_async(),
            group="ai-analytics",
            exclusive=True,
        )

    @on(DataTable.RowSelected, "#activities-table")
    @on(DataTable.RowHighlighted, "#activities-table")
    async def _on_activity_row_selected(
        self, event: DataTable.RowSelected | DataTable.RowHighlighted
    ) -> None:
        idx = event.cursor_row
        if 0 <= idx < len(self._activities_cache):
            row = self._activities_cache[idx]
            await self.query_one("#activity-detail", Markdown).update(
                self._format_activity_detail_markdown(row)
            )

    @on(Button.Pressed, "#btn-refresh-calendar")
    async def _on_refresh_calendar(self) -> None:
        await self._refresh_calendar_and_library_async()

    @on(Button.Pressed, "#btn-goto-plan")
    def _on_goto_plan(self) -> None:
        self.action_quick_plan()

    @on(DataTable.RowSelected, "#calendar-table")
    async def _on_calendar_row_selected(self, event: DataTable.RowSelected) -> None:
        idx = event.cursor_row
        if 0 <= idx < len(self._calendar_events_cache):
            ev = self._calendar_events_cache[idx]
            desc = ev.get("description") or "*No structured steps.*"
            detail_md = (
                f"**{ev.get('name', 'Event')}** (`{str(ev.get('start_date_local') or '')[:10]}` · `{ev.get('type', '—')}`)\n\n"
                f"```text\n{desc}\n```"
            )
            await self.query_one("#calendar-event-detail", Markdown).update(detail_md)

    @on(Select.Changed, "#evidence-topic-select")
    def _on_evidence_topic_changed(self, event: Select.Changed) -> None:
        topic = str(event.value or "all")
        self._load_curated_evidence(topic)

    @on(Button.Pressed, "#btn-curated-evidence")
    def _on_curated_evidence_pressed(self) -> None:
        topic = str(self.query_one("#evidence-topic-select", Select).value or "all")
        self._load_curated_evidence(topic)

    @on(Button.Pressed, "#btn-search-evidence")
    async def _on_search_evidence_pressed(self) -> None:
        query = self.query_one("#evidence-query-input", Input).value
        await self._run_evidence_lookup_async(query=query)

    @on(Input.Submitted, "#evidence-query-input")
    async def _on_evidence_query_submitted(self, event: Input.Submitted) -> None:
        await self._run_evidence_lookup_async(query=event.value)


def run_tui(graph: Any = None) -> None:
    """Launch the full-screen Textual TUI for Kinesis."""
    app = KinesisTUI(graph=graph)
    app.run()
