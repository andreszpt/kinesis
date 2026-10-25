from __future__ import annotations

import asyncio
import json
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.events import Resize
from textual.message import Message
from textual.theme import Theme
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    Markdown,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
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
    "Ask anything, or use `/plan <workout>`, `/analyze [days]`, `/evidence [query]`, `/status`, `/clear`.\n\n"
    "*Enter sends · Shift+Enter (or Ctrl+J) adds a line.*"
)

CURATED_CATEGORY_LABELS = {
    "concurrent_interference": "Concurrent Training Interference",
    "running_economy_and_strength": "Running Economy & Strength",
    "intensity_distribution_and_polarization": "Intensity Distribution & Polarization",
    "hrv_and_load_monitoring": "HRV & Load Monitoring",
}

_PALETTE = {
    "base": "#1e1e2e",
    "panel": "#181825",
    "surface": "#313244",
    "raised": "#45475a",
    "text": "#cdd6f4",
    "muted": "#a6adc8",
    "line": "#7f849c",
    "focus": "#b4befe",
    "accent": "#89b4fa",
    "status": "#a6e3a1",
    "warn": "#f9e2af",
    "danger": "#f38ba8",
}


def _fmt(value: Any, suffix: str = "", empty: str = "—") -> str:
    """Render a value without ever leaking ``None`` into the UI."""
    if value is None or value == "":
        return empty
    return f"{value}{suffix}"


def _whole(value: Any, empty: str = "—") -> str:
    """Render a number as a whole number for the compact sidebar."""
    if value is None or value == "":
        return empty
    try:
        return f"{round(float(value))}"
    except (TypeError, ValueError):
        return str(value)


def _short_date(value: Any) -> str:
    """Return a consistent MM-DD date for compact tables."""
    text = str(value or "")
    if len(text) >= 10:
        return text[5:10]
    return text or "—"


def _tsb_state(tsb: Any) -> str:
    if tsb is None:
        return "—"
    try:
        value = float(tsb)
    except (TypeError, ValueError):
        return "—"
    if value >= 5:
        return "✓ fresh"
    if value >= -10:
        return "✓ steady"
    if value >= -25:
        return "▲ strain"
    return "✗ over"


def _acwr_state(acwr: Any) -> str:
    if acwr is None:
        return "—"
    try:
        value = float(acwr)
    except (TypeError, ValueError):
        return "—"
    if 0.8 <= value <= 1.3:
        return "✓ optimal"
    if value < 0.8:
        return "▲ under"
    if value <= 1.5:
        return "▲ caution"
    return "✗ high"


def _alerts_state(count: Any) -> str:
    if count is None:
        return "—"
    try:
        value = int(count)
    except (TypeError, ValueError):
        return "—"
    if value == 0:
        return "✓ clear"
    if value <= 2:
        return "▲ watch"
    return "✗ act"


def _clock_now(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%H:%M")


def _table_cell(
    value: Any, *, width: int | None = None, right: bool = False
) -> Text:
    """A single-line table cell that ellipsizes instead of hard-truncating."""
    text = str(value if value is not None and value != "" else "—")
    if right and width is not None and len(text) < width:
        text = text.rjust(width)
    cell = Text(text)
    cell.no_wrap = True
    cell.overflow = "ellipsis"
    return cell


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
        return (
            f"**Failed (`{pending.action}`)**: "
            f"{_fmt(result.get('error'), empty='Unknown error')}"
        )
    if status == "uploaded":
        events = result.get("events") or []
        lines = [
            f"**Uploaded {_fmt(result.get('uploaded_count', len(events)))} Workout(s) to Intervals.icu**",
        ]
        for ev in events[:10]:
            if isinstance(ev, dict):
                date_part = str(ev.get("start_date_local") or "")[:10] or "—"
                lines.append(
                    f"- `{date_part}` **{ev.get('name') or 'Workout'}** "
                    f"(`{ev.get('type') or '—'}`)"
                )
        return "\n".join(lines)
    if status == "deleted":
        return f"**Deleted Event** `{_fmt(result.get('event_id'))}` — {pending.summary}"
    if status == "updated":
        return f"**Updated Event** `{_fmt(result.get('event_id'))}` — {pending.summary}"
    if status == "saved":
        return f"**Saved {_fmt(result.get('saved_count', 0))} Workout(s)** to Library"
    if status == "applied":
        return (
            f"**Applied Plan** `{_fmt(result.get('folder_id'))}` "
            f"starting `{_fmt(result.get('start_date_local'))}`"
        )
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
            f"Athlete: **{profile.get('name') or 'Athlete'}** (`{_fmt(profile.get('icu_resting_hr'))} bpm`)"
        )
    elif error:
        lines.append(f"Intervals.icu: {error}")
    elif not settings.has_intervals_credentials:
        lines.append("Intervals.icu: *No API key*")
    return "\n\n".join(lines)


def format_readiness_sidebar_markdown(
    report: dict[str, Any], as_of: str | None = None
) -> str:
    """Render compact readiness and polarization metrics for the sidebar.

    Values are laid out one metric per row in a monospace block so the sidebar
    never collapses them into a single wrapped paragraph.
    """
    if not report or "error" in report:
        err = report.get("error", "No data") if isinstance(report, dict) else "No data"
        return f"**Readiness Snapshot**\n\n*{err}*"

    lf = report.get("load_and_form") or {}
    rw = report.get("recovery_and_wellness") or {}
    rs = report.get("running_summary") or {}
    ss = report.get("strength_summary") or {}
    audit = report.get("concurrent_interference_audit") or {}
    dist = rs.get("intensity_distribution_3_zone") or {}
    hrv = rw.get("hrv") or {}
    rhr = rw.get("resting_hr") or {}
    sleep = rw.get("sleep_and_subjective") or {}

    days = report.get("analysis_window_days") or 28
    hrv_mean = hrv.get("rolling_7d_mean")
    hrv_base = hrv.get("baseline_mean")
    has_hrv = hrv_mean is not None or hrv_base is not None
    hrv_value = f"{_fmt(hrv_mean)}/{_fmt(hrv_base)} ms" if has_hrv else "—"
    zones = (
        f"{_whole(dist.get('zone_1_2_low_pct', 0))}/"
        f"{_whole(dist.get('zone_3_moderate_grey_zone_pct', 0))}/"
        f"{_whole(dist.get('zone_4_5_high_pct', 0))}%"
    )
    strength = (
        f"{_fmt(ss.get('total_sessions', 0))} "
        f"({_whole(ss.get('sessions_per_week', 0))}/wk)"
    )

    rows = [
        ("CTL", _fmt(lf.get("ctl_fitness")), ""),
        ("ATL", _fmt(lf.get("atl_fatigue")), ""),
        ("TSB", _fmt(lf.get("tsb_form")), _tsb_state(lf.get("tsb_form"))),
        ("ACWR", _fmt(lf.get("acwr_ratio")), _acwr_state(lf.get("acwr_ratio"))),
        ("HRV", hrv_value, "" if has_hrv else "no data"),
        ("RHR", _fmt(rhr.get("latest_rhr_bpm"), " bpm"), ""),
        ("Sleep", _fmt(sleep.get("rolling_7d_avg_hours"), " h"), ""),
        ("Run", _fmt(rs.get("weekly_avg_distance_km"), " km/wk"), ""),
        ("Zones", zones, ""),
        ("Strength", strength, ""),
        ("Alerts", _fmt(audit.get("warnings_count", 0)), _alerts_state(audit.get("warnings_count", 0))),
    ]
    body = "\n".join(
        f"{label:<8}{value:>10} {state}".rstrip() for label, value, state in rows
    )
    timestamp = as_of or _clock_now()
    md = (
        f"**Readiness ({days}d)** · as of {timestamp}\n\n"
        "```text\n"
        f"{body}\n"
        "```"
    )
    if not has_hrv:
        md += (
            "\n\n*No HRV data in this window — readiness uses load, RHR and sleep only.*"
        )
    return md


def format_analytics_dashboard_markdown(
    report: dict[str, Any], as_of: str | None = None
) -> str:
    """Render quantitative analysis report for the Analytics tab and `/analyze`."""
    if not report or "error" in report:
        err = report.get("error", "Intervals.icu credentials not configured.")
        return f"### Analytics Unavailable\n{err}"

    days = report.get("analysis_window_days") or 28
    lf = report.get("load_and_form") or {}
    rw = report.get("recovery_and_wellness") or {}
    rs = report.get("running_summary") or {}
    ss = report.get("strength_summary") or {}
    audit = report.get("concurrent_interference_audit") or {}
    dist = rs.get("intensity_distribution_3_zone") or {}
    hrv = rw.get("hrv") or {}
    rhr = rw.get("resting_hr") or {}
    sleep = rw.get("sleep_and_subjective") or {}
    warnings = audit.get("warnings_count", 0)
    timestamp = as_of or _clock_now()

    md = (
        f"### Load & Form ({days}d) · as of {timestamp}\n"
        f"- **CTL**: `{_fmt(lf.get('ctl_fitness'))}` · **ATL**: `{_fmt(lf.get('atl_fatigue'))}` · "
        f"**TSB**: `{_fmt(lf.get('tsb_form'))}` *{_tsb_state(lf.get('tsb_form'))}*\n"
        f"- **ACWR**: `{_fmt(lf.get('acwr_ratio'))}` *{_acwr_state(lf.get('acwr_ratio'))}* — "
        f"{_fmt(lf.get('acwr_interpretation'))}\n\n"
        f"### Autonomic Readiness\n"
        f"- **HRV**: `{_fmt(hrv.get('latest_hrv_rmssd'))} ms` "
        f"(7d `{_fmt(hrv.get('rolling_7d_mean'))}` vs base `{_fmt(hrv.get('baseline_mean'))}`, "
        f"CV `{_fmt(hrv.get('rolling_7d_cv_pct'))}%`)\n"
        f"- **RHR**: `{_fmt(rhr.get('latest_rhr_bpm'))} bpm` (7d `{_fmt(rhr.get('rolling_7d_mean_bpm'))}`) · "
        f"**Sleep**: `{_fmt(sleep.get('rolling_7d_avg_hours'))} h`\n\n"
        f"### Running & Polarization\n"
        f"- **Volume**: `{_fmt(rs.get('weekly_avg_distance_km'))} km/wk` across "
        f"`{_fmt(rs.get('total_sessions'))}` sessions (`{_fmt(rs.get('sessions_per_week'))}/wk`)\n"
        f"- **Zones**: Low `{_fmt(dist.get('zone_1_2_low_pct'))}%` · "
        f"Mod `{_fmt(dist.get('zone_3_moderate_grey_zone_pct'))}%` · "
        f"High `{_fmt(dist.get('zone_4_5_high_pct'))}%` (PI `{_fmt(dist.get('polarization_index'))}`)\n\n"
        f"### Strength & Spacing\n"
        f"- **Sessions**: `{_fmt(ss.get('total_sessions'))}` "
        f"(`{_fmt(ss.get('sessions_per_week'))}/wk`, `{_fmt(ss.get('total_duration_hours'))} h`)\n"
        f"- **Interference Alerts**: `{_fmt(warnings)}` *{_alerts_state(warnings)}*"
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

    lines = [f"**Upcoming (14d)** · {len(events)} planned\n"]
    for ev in events[:6]:
        date_str = _short_date(ev.get("start_date_local"))
        sport = ev.get("type") or ev.get("category") or "—"
        name = ev.get("name") or "Untitled"
        lines.append(f"`{date_str}` **{name}** *({sport})*")
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
    total_workouts = len(workouts)
    shown_workouts = workouts[:15]
    lines = [
        "### Library",
        f"**{len(folders)} {'folder' if len(folders) == 1 else 'folders'} · "
        f"{total_workouts} {'workout' if total_workouts == 1 else 'workouts'}**",
    ]
    if plan_data and isinstance(plan_data, dict) and not plan_data.get("error"):
        plan_name = (
            plan_data.get("training_plan_alias")
            or plan_data.get("name")
            or plan_data.get("id")
        )
        if plan_name:
            lines.append(f"Active plan: **{plan_name}**")

    if folders:
        lines.append("\n#### Folders")
        for f in folders[:10]:
            name = f.get("name") or "Untitled folder"
            count = _fmt(f.get("num_workouts", 0))
            folder_id = f.get("id")
            suffix = f" · `#{folder_id}`" if folder_id is not None else ""
            lines.append(f"- **{name}** — {count} workouts{suffix}")

    if workouts:
        lines.append(f"\n#### Workouts · showing {len(shown_workouts)} of {total_workouts}")
        for w in shown_workouts:
            name = w.get("name") or "Untitled workout"
            sport = w.get("type") or "Workout"
            mins = (
                f"{round(float(w['moving_time']) / 60)} min"
                if w.get("moving_time")
                else "— min"
            )
            lines.append(f"- **{name}** *({sport} · {mins})*")
        if total_workouts > len(shown_workouts):
            lines.append(
                f"\n*+{total_workouts - len(shown_workouts)} more workouts in the library.*"
            )
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
            title = p.get("title") or "Untitled"
            authors = p.get("authors") or "N/A"
            year = _fmt(p.get("year"), empty="n.d.")
            journal = p.get("journal") or "N/A"
            pmid = p.get("pmid")
            doi = p.get("doi")
            identifiers = []
            if pmid:
                identifiers.append(f"PMID `{pmid}`")
            if doi:
                identifiers.append(f"DOI `{doi}`")
            summary = str(p.get("abstract") or "").strip().replace("\n", " ")
            if len(summary) > 320:
                summary = summary[:320].rstrip() + "…"
            lines.append(
                f"#### {idx}. {title}\n\n"
                f"{authors} · {year} · *{journal}*\n\n"
                f"{' · '.join(identifiers) if identifiers else 'No identifier'}\n\n"
                f"{summary or '*No abstract available.*'}\n"
            )
        return "\n".join(lines)

    if not curated:
        return "*No evidence loaded.*"

    lines = ["### Curated Sports Science Evidence\n"]
    for cat, citations in curated.items():
        label = CURATED_CATEGORY_LABELS.get(cat, cat.replace("_", " ").title())
        lines.append(f"#### {label}")
        for c in citations:
            title = c.get("title") or "Untitled"
            authors = c.get("authors") or "N/A"
            year = _fmt(c.get("year"), empty="n.d.")
            journal = c.get("journal") or "N/A"
            pmid = c.get("pmid")
            application = c.get("practical_application") or "No application note."
            lines.append(
                f"**{title}**\n\n"
                f"{authors} · {year} · *{journal}*\n\n"
                f"PMID `{_fmt(pmid)}`\n\n"
                f"*Application:* {application}\n"
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


class StatusOverlay(Static):
    """Full-screen overlay used when the terminal is below the minimum size."""

    can_focus = True


class ChatTextArea(TextArea):
    """Multiline chat composer: Enter sends, Shift+Enter inserts a newline."""

    class Submitted(Message):
        """Posted when the user presses Enter to send the composer contents."""

        def __init__(self, input: ChatTextArea, text: str) -> None:
            super().__init__()
            self.input = input
            self.text = text

        @property
        def control(self) -> ChatTextArea:
            """The composer that posted this message."""
            return self.input

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self, self.text))
            return
        if event.key in ("shift+enter", "ctrl+j", "alt+enter"):
            event.stop()
            event.prevent_default()
            result = self.replace(
                "\n", *self.selection, maintain_selection_offset=False
            )
            if result is not None:
                self.move_cursor(result.end_location)
            return
        await super()._on_key(event)


class ConfirmationBanner(Container):
    """Inline confirmation bar shown when a mutating tool needs approval."""

    can_focus = True

    BINDINGS = [
        Binding("a", "approve", "Approve", show=False),
        Binding("r", "reject", "Reject", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Static("", id="confirmation-summary")
        with Horizontal(id="confirmation-actions"):
            yield Button("Approve", id="btn-approve-confirm", variant="success")
            yield Button("Reject", id="btn-reject-confirm", variant="error")
            yield Static("a approve · r reject", id="confirmation-hint")

    def action_approve(self) -> None:
        self.app.run_worker(self.app.approve_pending_confirmations())

    def action_reject(self) -> None:
        self.app.run_worker(self.app.reject_pending_confirmations())


KINESIS_THEME = Theme(
    name="kinesis",
    primary=_PALETTE["accent"],
    secondary=_PALETTE["focus"],
    accent=_PALETTE["status"],
    foreground=_PALETTE["text"],
    background=_PALETTE["base"],
    surface=_PALETTE["surface"],
    panel=_PALETTE["panel"],
    warning=_PALETTE["warn"],
    error=_PALETTE["danger"],
    success=_PALETTE["status"],
    dark=True,
    variables={
        "k-base": _PALETTE["base"],
        "k-panel": _PALETTE["panel"],
        "k-surface": _PALETTE["surface"],
        "k-raised": _PALETTE["raised"],
        "k-text": _PALETTE["text"],
        "k-muted": _PALETTE["muted"],
        "k-line": _PALETTE["line"],
        "k-focus": _PALETTE["focus"],
        "k-accent": _PALETTE["accent"],
        "k-status": _PALETTE["status"],
        "k-warn": _PALETTE["warn"],
        "k-danger": _PALETTE["danger"],
    },
)


class KinesisTUI(App[None]):
    """Minimal full-screen Textual TUI: coach session with a pulse rail."""

    TITLE = "kinesis"

    CSS = """
    Screen {
        background: $k-base;
        color: $k-text;
    }

    #top-bar {
        height: 1;
        margin: 0;
        padding: 0 2;
        align: left middle;
        background: $k-panel;
        color: $k-muted;
    }

    #top-brand {
        width: auto;
        height: 1;
        content-align: left middle;
        color: $k-accent;
        text-style: bold;
    }

    #top-telemetry {
        width: 1fr;
        height: 1;
        content-align: right middle;
        color: $k-muted;
        text-overflow: ellipsis;
    }

    #main-tabs {
        height: 1fr;
        background: $k-base;
    }

    Tabs {
        height: 2;
        padding: 0 2;
        background: $k-panel;
        border: none;
    }

    Tab {
        height: 1;
        padding: 0 3;
        content-align: center middle;
        color: $k-muted;
        background: transparent;
    }

    Tab:hover {
        color: $k-text;
        background: $k-raised;
    }

    Tab.-active {
        color: $k-accent;
        text-style: bold underline;
        background: $k-surface;
    }

    Underline {
        height: 1;
    }

    Underline > .underline--bar {
        color: $k-accent;
        background: $k-panel;
    }

    TabPane {
        padding: 0;
        background: $k-base;
    }

    #chat-workspace {
        height: 1fr;
    }

    #chat-main {
        width: 1fr;
        height: 1fr;
        padding: 0 2;
    }

    #chat-scroll {
        height: 1fr;
        padding: 0;
        border: none;
        background: $k-base;
    }

    .chat-card {
        height: auto;
        margin: 0 0 1 0;
        padding: 0 0 0 2;
        border: none;
        border-left: solid $k-accent;
        background: transparent;
    }

    .chat-card-title {
        text-style: bold;
        color: $k-muted;
        margin-bottom: 0;
    }

    .role-user { border-left: solid $k-text; }
    .role-user .chat-card-title { color: $k-text; }

    .agent-analyst { border-left: solid $k-focus; }
    .agent-analyst .chat-card-title { color: $k-focus; }

    .agent-researcher { border-left: solid $k-status; }
    .agent-researcher .chat-card-title { color: $k-status; }

    .agent-planner { border-left: solid $k-accent; }
    .agent-planner .chat-card-title { color: $k-accent; }

    .role-system { border-left: solid $k-line; }
    .role-system .chat-card-title { color: $k-muted; }

    .tool-strip {
        height: auto;
        color: $k-muted;
        padding: 0 0 0 2;
        margin: 0 0 1 0;
    }

    #agent-activity-bar {
        height: 1;
        padding: 0 1;
        margin: 0;
        color: $k-muted;
        text-overflow: ellipsis;
    }

    #confirmation-bar {
        display: none;
        height: auto;
        max-height: 3;
        margin: 0 0 1 0;
        padding: 0 1;
        border-left: solid $k-warn;
        background: $k-surface;
    }

    #confirmation-bar.visible {
        display: block;
    }

    #confirmation-bar:focus {
        border-left: solid $k-focus;
    }

    #confirmation-summary {
        width: 1fr;
        height: 2;
        color: $k-text;
        text-overflow: ellipsis;
    }

    #confirmation-actions {
        height: 1;
        align: left middle;
    }

    #confirmation-actions Button {
        min-width: 9;
        height: 1;
        margin-right: 1;
    }

    #confirmation-hint {
        width: auto;
        height: 1;
        color: $k-muted;
        content-align: left middle;
    }

    Button {
        min-width: 8;
        height: 1;
        padding: 0 1;
        background: $k-surface;
        color: $k-text;
        border: none;
    }

    Button:hover, Button:focus {
        background: $k-raised;
        color: $k-text;
    }

    Button:disabled {
        background: $k-panel;
        color: $k-muted;
    }

    Button.-primary {
        background: $k-accent;
        color: $k-base;
        text-style: bold;
    }

    Button.-success {
        background: $k-status;
        color: $k-base;
        text-style: bold;
    }

    Button.-error {
        background: transparent;
        color: $k-danger;
    }

    Button.-error:hover, Button.-error:focus {
        background: $k-danger;
        color: $k-base;
    }

    #chat-input-bar {
        height: auto;
        min-height: 3;
        padding: 0;
        align: left middle;
    }

    #mode-chip {
        width: 7;
        height: 3;
        margin: 0 1 0 0;
        content-align: center middle;
        color: $k-muted;
        background: $k-surface;
    }

    #mode-chip.plan {
        color: $k-base;
        background: $k-accent;
        text-style: bold;
    }

    #chat-input {
        width: 1fr;
        height: auto;
        min-height: 3;
        max-height: 7;
        padding: 0 2;
        border: round $k-line;
        background: $k-surface;
        color: $k-text;
    }

    #chat-input .text-area--placeholder {
        color: $k-muted;
    }

    #evidence-query-input {
        width: 1fr;
        height: 3;
        padding: 0 2;
        border: round $k-line;
        background: $k-surface;
        color: $k-text;
    }

    #chat-input:focus, #evidence-query-input:focus {
        border: round $k-focus;
        background: $k-surface;
    }

    #sidebar {
        width: 36;
        height: 1fr;
        border-left: solid $k-line;
        padding: 1 2;
        background: $k-panel;
    }

    #sidebar.hidden {
        display: none;
    }

    .sidebar-card {
        height: auto;
        margin: 0 0 1 0;
        padding: 0 0 1 0;
        border: none;
        border-bottom: solid $k-line;
        background: transparent;
    }

    #sidebar-pipeline {
        color: $k-muted;
    }

    #sidebar-readiness MarkdownFence {
        margin: 0;
        padding: 0;
        background: transparent;
    }

    #sidebar-readiness MarkdownFence > Label {
        padding: 0;
        color: $k-text;
    }

    .tab-toolbar {
        height: 2;
        padding: 0 2;
        align: left middle;
        background: $k-panel;
        border-bottom: solid $k-line;
    }

    .tab-toolbar Button {
        margin-right: 1;
    }

    .tab-toolbar Button.day-btn {
        min-width: 5;
    }

    .tab-toolbar Select, .tab-toolbar Input {
        height: 1;
        border: none;
        background: $k-surface;
        color: $k-text;
        padding: 0 1;
        margin-right: 1;
    }

    .tab-toolbar Input#evidence-query-input {
        height: 1;
        border: none;
        background: $k-surface;
        padding: 0 1;
    }

    #analytics-status {
        width: 1fr;
        height: 1;
        content-align: right middle;
        color: $k-muted;
        padding: 0 1;
    }

    #analytics-body, #calendar-body {
        height: 1fr;
        padding: 1 2;
    }

    #analytics-report-scroll {
        width: 1fr;
        height: 1fr;
        padding: 0 2 0 0;
    }

    #analytics-ai-markdown {
        display: none;
        height: auto;
        margin: 0 0 1 0;
        padding: 1 2;
        border-left: solid $k-focus;
        background: $k-surface;
    }

    #analytics-ai-markdown.visible {
        display: block;
    }

    #analytics-activities-pane {
        width: 1fr;
        height: 1fr;
        border-left: solid $k-line;
        padding: 0 0 0 2;
    }

    #activities-table {
        height: 1fr;
        margin-bottom: 1;
    }

    #activity-detail-scroll {
        width: 1fr;
        height: 9;
        border-top: solid $k-line;
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

    #calendar-detail-scroll {
        width: 1fr;
        height: 10;
        border-top: solid $k-line;
        padding: 1 0;
    }

    #library-pane {
        width: 2fr;
        height: 1fr;
        border-left: solid $k-line;
        padding: 0 0 0 2;
    }

    .pane-hidden {
        display: none;
    }

    .pane-title {
        text-style: bold;
        color: $k-text;
        margin-bottom: 1;
    }

    #evidence-topic-select {
        width: 28;
    }

    Screen.narrow #evidence-topic-select {
        width: 16;
    }

    #evidence-scroll {
        height: 1fr;
        padding: 1 2;
    }

    DataTable {
        background: $k-base;
        color: $k-text;
        overflow-x: hidden;
    }

    DataTable > .datatable--header {
        background: $k-surface;
        color: $k-text;
        text-style: bold;
    }

    DataTable > .datatable--cursor {
        background: $k-raised;
        color: $k-text;
    }

    #bottom-bar {
        height: 1;
        padding: 0 2;
        align: left middle;
        background: $k-panel;
        color: $k-muted;
    }

    #bottom-hints {
        width: 1fr;
        height: 1;
        content-align: left middle;
        color: $k-muted;
        text-overflow: ellipsis;
    }

    #bottom-keys {
        width: auto;
        height: 1;
        padding-left: 2;
        content-align: right middle;
        color: $k-muted;
    }

    #too-small {
        display: none;
        width: 1fr;
        height: 1fr;
        content-align: center middle;
        background: $k-base;
        color: $k-text;
    }

    Screen.too-small #top-bar,
    Screen.too-small #main-tabs,
    Screen.too-small #bottom-bar {
        display: none;
    }

    Screen.too-small #too-small {
        display: block;
    }
    """

    BINDINGS = [
        Binding("ctrl+b", "toggle_sidebar", "Sidebar", show=True),
        Binding("ctrl+r", "refresh_data", "Refresh", show=True),
        Binding("ctrl+l", "clear_chat", "Clear", show=True),
        Binding("1", "tab_chat", "Chat", show=False),
        Binding("2", "tab_analytics", "Analytics", show=False),
        Binding("3", "tab_calendar", "Calendar", show=False),
        Binding("4", "tab_evidence", "Evidence", show=False),
        Binding("ctrl+1", "tab_chat", "Chat", show=False),
        Binding("ctrl+2", "tab_analytics", "Analytics", show=False),
        Binding("ctrl+3", "tab_calendar", "Calendar", show=False),
        Binding("ctrl+4", "tab_evidence", "Evidence", show=False),
        Binding("f1", "tab_chat", "Chat", show=False),
        Binding("f2", "tab_analytics", "Analytics", show=False),
        Binding("f3", "tab_calendar", "Calendar", show=False),
        Binding("f4", "tab_evidence", "Evidence", show=False),
        Binding("ctrl+p", "quick_plan", "Plan", show=False),
        Binding("escape", "stop_turn", "Stop", show=False),
        Binding("q", "quit_small", "Quit", show=False),
        Binding("ctrl+q", "quit", "Quit", show=True),
    ]

    def __init__(
        self,
        graph: Any = None,
        auto_refresh_on_mount: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.register_theme(KINESIS_THEME)
        self.theme = "kinesis"
        self._graph = graph
        self._auto_refresh_on_mount = auto_refresh_on_mount
        self.conversation_messages: list[AnyMessage] = []
        self.pending_confirmations: list[PendingConfirmation] = []
        self.analytics_days: int = 28
        self._activities_cache: list[dict[str, Any]] = []
        self._calendar_events_cache: list[dict[str, Any]] = []
        self._activities_name_key: Any = None
        self._calendar_name_key: Any = None
        self._is_busy: bool = False
        self._mode: str = "auto"
        self._queued_messages: list[str] = []
        self._evidence_initialized: bool = False
        self._sidebar_override: bool | None = None
        self._is_too_small: bool = False
        self._thinking_started: float = 0.0
        self._thinking_timer: Any = None
        self._busy_task: asyncio.Task[Any] | None = None

    def _get_graph(self) -> Any:
        if self._graph is None:
            self._graph = build_hybrid_training_graph()
        return self._graph

    def compose(self) -> ComposeResult:
        with Horizontal(id="top-bar"):
            yield Static("// kinesis", id="top-brand")
            yield Static("syncing...", id="top-telemetry")

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
                            yield Static("auto", id="mode-chip")
                            yield ChatTextArea(
                                placeholder="Ask anything",
                                id="chat-input",
                                show_line_numbers=False,
                                highlight_cursor_line=False,
                                soft_wrap=True,
                            )
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
                    yield Button("AI Commentary", id="btn-ai-analytics")
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
                    yield Button("+ Plan Workout", id="btn-goto-plan")
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
                        compact=True,
                        id="evidence-topic-select",
                    )
                    yield Input(
                        placeholder="Search PubMed",
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
            yield Static("1-4 tabs", id="bottom-hints")
            yield Static("^b sidebar   ^r refresh   ^l clear   ^q quit", id="bottom-keys")

        yield StatusOverlay("", id="too-small")

    async def on_mount(self) -> None:
        activities_table = self.query_one("#activities-table", DataTable)
        for label, width in (
            ("Date", 5),
            ("Sport", 7),
            ("Name", 20),
            ("km", 4),
            ("Min", 3),
            ("Load", 4),
        ):
            key = activities_table.add_column(label, width=width)
            if label == "Name":
                self._activities_name_key = key

        calendar_table = self.query_one("#calendar-table", DataTable)
        for label, width in (
            ("Date", 5),
            ("Type", 7),
            ("Name", 20),
            ("Min", 4),
            ("Load", 5),
        ):
            key = calendar_table.add_column(label, width=width)
            if label == "Name":
                self._calendar_name_key = key

        self.query_one("#chat-input", ChatTextArea).focus()
        self._apply_layout()

        if self._auto_refresh_on_mount:
            self.run_worker(self._refresh_all_data_async(), exclusive=False)

    # ------------------------------------------------------------------
    # Layout, Modes & Busy State
    # ------------------------------------------------------------------
    def _apply_layout(self) -> None:
        """Apply width breakpoints and the too-small state."""
        try:
            width, height = self.size.width, self.size.height
        except Exception:  # noqa: BLE001 - called before the screen is ready
            return
        if width <= 0 or height <= 0:
            return

        was_too_small = self._is_too_small
        self._is_too_small = width < 60 or height < 24
        self.screen.set_class(self._is_too_small, "too-small")
        self.screen.set_class(width < 80, "narrow")
        if self._is_too_small and not was_too_small:
            try:
                self.call_after_refresh(
                    self.query_one("#too-small", StatusOverlay).focus
                )
            except Exception:  # noqa: BLE001
                pass
        elif was_too_small and not self._is_too_small:
            try:
                self.query_one("#chat-input", ChatTextArea).focus()
            except Exception:  # noqa: BLE001
                pass

        try:
            sidebar = self.query_one("#sidebar", VerticalScroll)
            show_sidebar = (
                self._sidebar_override
                if self._sidebar_override is not None
                else width >= 100
            )
            sidebar.set_class(not show_sidebar, "hidden")

            one_pane = width < 100
            self.query_one("#analytics-activities-pane").set_class(
                one_pane, "pane-hidden"
            )
            self.query_one("#library-pane").set_class(one_pane, "pane-hidden")

            self.query_one("#too-small", Static).update(
                f"current {width}×{height} · needs 80×24 · q to quit"
            )
        except Exception:  # noqa: BLE001 - before compose has finished
            return

        if self.pending_confirmations:
            self._render_confirmation_summary()

        self.call_after_refresh(self._fit_table_columns)

    def _fit_table_columns(self) -> None:
        """Size the name column to the space left after fixed-width metrics."""
        try:
            activities = self.query_one("#activities-table", DataTable)
            available = activities.size.width
            if available > 0 and self._activities_name_key is not None:
                fixed = 5 + 7 + 4 + 3 + 4 + 2 * 6 + 2
                new_width = max(6, available - fixed)
                column = activities.columns[self._activities_name_key]
                if column.width != new_width:
                    column.width = new_width
                    activities._require_update_dimensions = True
                    activities._clear_caches()
                    activities.refresh(layout=True)
        except Exception:  # noqa: BLE001
            pass

        try:
            calendar = self.query_one("#calendar-table", DataTable)
            available = calendar.size.width
            if available > 0 and self._calendar_name_key is not None:
                fixed = 5 + 7 + 4 + 5 + 2 * 5 + 2
                new_width = max(6, available - fixed)
                column = calendar.columns[self._calendar_name_key]
                if column.width != new_width:
                    column.width = new_width
                    calendar._require_update_dimensions = True
                    calendar._clear_caches()
                    calendar.refresh(layout=True)
        except Exception:  # noqa: BLE001
            pass

    @on(TabbedContent.TabActivated)
    def _on_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        active = event.tabbed_content.active
        if active in ("tab-analytics", "tab-calendar"):
            self.call_after_refresh(self._fit_table_columns)
        if active == "tab-evidence" and not self._evidence_initialized:
            self._load_curated_evidence("all")

    def on_resize(self, event: Resize) -> None:
        self._apply_layout()

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        try:
            chip = self.query_one("#mode-chip", Static)
            chip.update(mode)
            chip.set_class(mode == "plan", "plan")
        except Exception:  # noqa: BLE001
            pass

    def _thinking_status_text(self) -> str:
        elapsed = int(max(0.0, time.monotonic() - self._thinking_started))
        text = f"Thinking {elapsed}s · esc stop"
        if self._queued_messages:
            text += f" · {len(self._queued_messages)} queued"
        return text

    def _tick_thinking(self) -> None:
        if self._is_busy:
            self._set_activity_status(self._thinking_status_text())

    def _start_thinking(self) -> None:
        self._thinking_started = time.monotonic()
        self._tick_thinking()
        if self._thinking_timer is None:
            self._thinking_timer = self.set_interval(1.0, self._tick_thinking)

    def _stop_thinking(self) -> None:
        if self._thinking_timer is not None:
            self._thinking_timer.stop()
            self._thinking_timer = None

    async def _drain_queue_async(self) -> None:
        if self._is_busy or not self._queued_messages:
            return
        await self.submit_user_input(self._queued_messages.pop(0))

    # ------------------------------------------------------------------
    # Keybinding Actions
    # ------------------------------------------------------------------
    def action_toggle_sidebar(self) -> None:
        sidebar = self.query_one("#sidebar", VerticalScroll)
        currently_visible = not sidebar.has_class("hidden")
        self._sidebar_override = not currently_visible
        self._apply_layout()

    def action_refresh_data(self) -> None:
        self.run_worker(self._refresh_all_data_async(), exclusive=False)

    def action_quick_plan(self) -> None:
        self.query_one("#main-tabs", TabbedContent).active = "tab-chat"
        self._set_mode("plan")
        chat_input = self.query_one("#chat-input", ChatTextArea)
        if not chat_input.text.startswith("/plan"):
            chat_input.load_text("/plan ")
            chat_input.move_cursor(chat_input.document.end)
        chat_input.focus()

    def action_stop_turn(self) -> None:
        if self._is_busy and self._busy_task is not None and not self._busy_task.done():
            self._set_activity_status("Stopping...")
            self._busy_task.cancel()

    def action_quit_small(self) -> None:
        if self._is_too_small:
            self.exit()

    async def action_clear_chat(self) -> None:
        await self._clear_chat_history()

    def action_tab_chat(self) -> None:
        self.query_one("#main-tabs", TabbedContent).active = "tab-chat"
        self.query_one("#chat-input", ChatTextArea).focus()

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
        self._evidence_initialized = True

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

        as_of = _clock_now()
        await self.query_one("#sidebar-readiness", Markdown).update(
            format_readiness_sidebar_markdown(report, as_of=as_of)
        )
        await self.query_one("#analytics-markdown", Markdown).update(
            format_analytics_dashboard_markdown(report, as_of=as_of)
        )

        if "load_and_form" in report:
            lf = report["load_and_form"]
            hrv = (
                report.get("recovery_and_wellness", {})
                .get("hrv", {})
                .get("rolling_7d_mean")
            )
            hrv_part = f" · HRV {_fmt(hrv)}ms" if hrv is not None else ""
            self.query_one("#top-telemetry", Static).update(
                f"CTL {_fmt(lf.get('ctl_fitness'))} · "
                f"ATL {_fmt(lf.get('atl_fatigue'))} · "
                f"TSB {_fmt(lf.get('tsb_form'))}{hrv_part} · as of {as_of}"
            )
        else:
            self.query_one("#top-telemetry", Static).update(
                "no Intervals.icu data · ctrl+r to retry"
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
                _table_cell(_short_date(merged.get("date"))),
                _table_cell(merged.get("type") or "—"),
                _table_cell(merged.get("name") or "Untitled"),
                _table_cell(
                    merged.get("distance_km")
                    if merged.get("distance_km") is not None
                    else "—",
                    width=4,
                    right=True,
                ),
                _table_cell(
                    merged.get("duration_min")
                    if merged.get("duration_min") is not None
                    else "—",
                    width=3,
                    right=True,
                ),
                _table_cell(
                    merged.get("load")
                    if merged.get("load") is not None
                    else "—",
                    width=4,
                    right=True,
                ),
            )

        self._activities_cache = enriched_rows
        if enriched_rows:
            await self.query_one("#activity-detail", Markdown).update(
                self._format_activity_detail_markdown(enriched_rows[0])
            )
            status_widget.update(
                f"Window: {days}d · {len(enriched_rows)} activities · {as_of}"
            )
        else:
            await self.query_one("#activity-detail", Markdown).update(
                "*No completed activities in this window.*"
            )
            status_widget.update(f"Window: {days}d · {as_of}")

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
                _table_cell(_short_date(ev.get("start_date_local"))),
                _table_cell(ev.get("type") or "—"),
                _table_cell(ev.get("name") or "Untitled"),
                _table_cell(mins, width=4, right=True),
                _table_cell(
                    ev.get("icu_training_load")
                    if ev.get("icu_training_load") is not None
                    else "—",
                    width=5,
                    right=True,
                ),
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
        pipe_str = " > ".join(pipeline) if pipeline else "auto"
        self.query_one("#sidebar-pipeline", Static).update(f"Route: {pipe_str}")

    def _render_confirmation_summary(self) -> None:
        """Wrap the confirmation summary into at most two ellipsized lines."""
        if not self.pending_confirmations:
            self.query_one("#confirmation-summary", Static).update("")
            return
        text = "Confirm: " + " | ".join(
            p.summary for p in self.pending_confirmations
        )
        width = 0
        try:
            width = self.query_one("#chat-main", Vertical).content_size.width - 3
        except Exception:  # noqa: BLE001
            pass
        if width <= 0:
            try:
                banner = self.query_one("#confirmation-bar", ConfirmationBanner)
                width = banner.content_size.width
            except Exception:  # noqa: BLE001
                width = 0
        if width <= 0:
            sidebar_visible = not self.query_one(
                "#sidebar", VerticalScroll
            ).has_class("hidden")
            width = self.size.width - (36 if sidebar_visible else 0) - 6
        width = max(16, width)
        lines = textwrap.wrap(text, width=width) or [text]
        if len(lines) > 2:
            lines = lines[:2]
            lines[-1] = lines[-1][: max(1, width - 1)].rstrip() + "…"
        self.query_one("#confirmation-summary", Static).update("\n".join(lines))

    def _update_confirmation_banner(self) -> None:
        banner = self.query_one("#confirmation-bar", ConfirmationBanner)
        if self.pending_confirmations:
            self._render_confirmation_summary()
            banner.add_class("visible")
            self.call_after_refresh(self._render_confirmation_summary)
            if not self._is_too_small:
                try:
                    banner.focus()
                except Exception:  # noqa: BLE001
                    pass
        else:
            self.query_one("#confirmation-summary", Static).update("")
            banner.remove_class("visible")
            if not self._is_too_small:
                try:
                    self.query_one("#chat-input", ChatTextArea).focus()
                except Exception:  # noqa: BLE001
                    pass

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
        self._queued_messages.clear()
        self._set_mode("auto")
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
        if not user_input:
            return
        if self._is_busy:
            self._queued_messages.append(user_input)
            self._set_activity_status(self._thinking_status_text())
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
            self._set_mode("auto")
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
            self._set_mode("auto")
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
            self._set_mode("auto")
            return

        selected_mode = self._mode
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
        self._busy_task = asyncio.current_task()
        self._start_thinking()
        try:
            await self._append_chat_card(
                title="you",
                markdown_text=prompt_text,
                role_class="role-user",
            )
            self.conversation_messages.append(HumanMessage(content=prompt_text))
            prev_len = len(self.conversation_messages)
            self._set_activity_status(self._thinking_status_text())

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
                self._set_activity_status("Awaiting confirmation · a approve · r reject")
            elif _pending_question(self.conversation_messages):
                self._set_activity_status("Awaiting your reply.")
            else:
                self._set_activity_status("")
        except asyncio.CancelledError:
            try:
                await self._append_chat_card(
                    title="system",
                    markdown_text="*Stopped.*",
                    role_class="role-system",
                )
            except Exception:  # noqa: BLE001 - app may be shutting down
                pass
            self._set_activity_status("Stopped.")
        except Exception as exc:  # noqa: BLE001
            await self._append_chat_card(
                title="error",
                markdown_text=f"`{exc}`",
                role_class="role-system",
            )
            self._set_activity_status("")
        finally:
            self._is_busy = False
            self._busy_task = None
            self._stop_thinking()
            self._set_mode("auto")
            if self.pending_confirmations:
                try:
                    self.query_one("#confirmation-bar", ConfirmationBanner).focus()
                except Exception:  # noqa: BLE001
                    pass
            elif not self._is_too_small:
                try:
                    self.query_one("#chat-input", ChatTextArea).focus()
                except Exception:  # noqa: BLE001
                    pass
            if self._queued_messages:
                self.run_worker(self._drain_queue_async(), group="chat", exclusive=True)

    async def approve_pending_confirmations(self) -> None:
        """Approve and execute all currently pending intervals.icu confirmations."""
        if not self.pending_confirmations or self._is_busy:
            return
        self._is_busy = True
        self._busy_task = asyncio.current_task()
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
        except asyncio.CancelledError:
            self._set_activity_status("Stopped.")
        except Exception as exc:  # noqa: BLE001
            await self._append_chat_card(
                title="error",
                markdown_text=f"`{exc}`",
                role_class="role-system",
            )
            self._set_activity_status("")
        finally:
            self._is_busy = False
            self._busy_task = None
            self._stop_thinking()
            try:
                self.query_one("#chat-input", ChatTextArea).focus()
            except Exception:  # noqa: BLE001
                pass
            if self._queued_messages:
                self.run_worker(self._drain_queue_async(), group="chat", exclusive=True)

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
        self._evidence_initialized = True
        return md

    # ------------------------------------------------------------------
    # UI Event Handlers
    # ------------------------------------------------------------------
    @on(ChatTextArea.Submitted, "#chat-input")
    async def _on_chat_input_submitted(self, event: ChatTextArea.Submitted) -> None:
        text = event.text.strip()
        event.input.clear()
        if not text:
            return
        if self._is_busy:
            self._queued_messages.append(text)
            self._set_activity_status(self._thinking_status_text())
            return
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
            event_id = ev.get("id")
            id_part = f" · ID `#{event_id}`" if event_id is not None else ""
            detail_md = (
                f"**{ev.get('name') or 'Event'}** "
                f"(`{str(ev.get('start_date_local') or '')[:10]}` · "
                f"`{ev.get('type') or '—'}`{id_part})\n\n"
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
