from __future__ import annotations

import asyncio
import io
import json
import os
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rich.console import Console
from textual.containers import VerticalScroll
from textual.widgets import DataTable, Input, TabbedContent

from hybrid_training_agent import cli
from hybrid_training_agent.clients.http_client import reset_shared_http_client
from hybrid_training_agent.clients.intervals_client import _intervals_cache
from hybrid_training_agent.clients.science_client import _science_cache
from hybrid_training_agent.tui import (
    ChatMessageCard,
    ConfirmationBanner,
    KinesisTUI,
    PendingConfirmation,
    StatusOverlay,
    execute_pending_confirmation,
    extract_pending_confirmations,
    format_analytics_dashboard_markdown,
    format_calendar_sidebar_markdown,
    format_confirmation_result_markdown,
    format_evidence_markdown,
    format_library_markdown,
    format_readiness_sidebar_markdown,
)

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


@pytest.fixture(autouse=True)
def _hermetic_tui_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERVALS_ICU_API_KEY", "")
    reset_shared_http_client()
    _intervals_cache.clear()
    _science_cache.clear()
    yield
    _intervals_cache.clear()
    _science_cache.clear()
    reset_shared_http_client()


def test_extract_pending_confirmations_and_execute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERVALS_ICU_API_KEY", "test_secret")
    workout_item = {
        "date": "2026-10-26",
        "name": "Easy Run",
        "sport_type": "Run",
        "description": "- 30m 70% Pace",
    }
    ai_call = AIMessage(
        content="Let me preview this workout.",
        tool_calls=[
            {
                "id": "call_upload_1",
                "name": "upload_training_plan_to_intervals",
                "args": {"workouts": [workout_item], "upsert": True, "confirm": False},
            }
        ],
    )
    tool_preview = ToolMessage(
        content=json.dumps(
            {
                "status": "confirmation_required",
                "action": "upload_training_plan_to_intervals",
                "summary": "Upload/update 1 workout(s) on the intervals.icu calendar (upsert=True).",
                "details": {"workouts": [workout_item], "upsert": True},
            }
        ),
        tool_call_id="call_upload_1",
        name="upload_training_plan_to_intervals",
    )
    ai_final = AIMessage(
        content="Here is the preview. Would you like me to upload it?",
        name="Hybrid_Coach_Planner",
    )

    pending = extract_pending_confirmations([ai_call, tool_preview, ai_final])
    assert len(pending) == 1
    assert pending[0].tool_name == "upload_training_plan_to_intervals"
    assert pending[0].tool_args["workouts"] == [workout_item]
    assert "Upload/update 1 workout(s)" in pending[0].summary

    with respx.mock(base_url="https://intervals.icu/api/v1") as respx_mock:
        bulk_route = respx_mock.post("/athlete/0/events/bulk").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "id": 999,
                        "name": "Easy Run",
                        "type": "Run",
                        "start_date_local": "2026-10-26T00:00:00",
                    }
                ],
            )
        )
        res = execute_pending_confirmation(pending[0])
        assert bulk_route.called
        assert res["status"] == "uploaded"
        md = format_confirmation_result_markdown(pending[0], res)
        assert "Uploaded 1 Workout(s)" in md
        assert "Easy Run" in md


def test_markdown_formatters_handle_data_and_errors() -> None:
    assert "Analytics Unavailable" in format_analytics_dashboard_markdown({"error": "Missing key"})
    assert "Readiness Snapshot" in format_readiness_sidebar_markdown({"error": "No key"})
    assert "No planned workouts" in format_calendar_sidebar_markdown({"events": []})
    assert "No peer-reviewed papers" in format_evidence_markdown(papers=[], query="nonexistent")

    sample_report = {
        "analysis_window_days": 28,
        "load_and_form": {
            "ctl_fitness": 52.0,
            "atl_fatigue": 48.0,
            "tsb_form": 4.0,
            "acwr_ratio": 0.92,
            "acwr_interpretation": "Optimal",
        },
        "recovery_and_wellness": {
            "hrv": {"latest_hrv_rmssd": 70.0, "rolling_7d_mean": 68.0, "baseline_mean": 67.0, "rolling_7d_cv_pct": 4.2},
            "resting_hr": {"latest_rhr_bpm": 46, "rolling_7d_mean_bpm": 47.0, "baseline_mean_bpm": 48.0},
            "sleep_and_subjective": {"rolling_7d_avg_hours": 7.8},
        },
        "running_summary": {
            "total_sessions": 12,
            "sessions_per_week": 3.0,
            "total_distance_km": 120.0,
            "weekly_avg_distance_km": 30.0,
            "intensity_distribution_3_zone": {
                "zone_1_2_low_pct": 80.0,
                "zone_3_moderate_grey_zone_pct": 5.0,
                "zone_4_5_high_pct": 15.0,
                "polarization_index": 2.03,
                "target_recommendation": "80/20",
            },
        },
        "strength_summary": {"total_sessions": 8, "sessions_per_week": 2.0, "total_duration_hours": 7.5},
        "concurrent_interference_audit": {"warnings_count": 1, "warnings": ["Same-day alert"]},
    }
    sidebar_md = format_readiness_sidebar_markdown(sample_report)
    assert "52.0" in sidebar_md and "30.0 km/wk" in sidebar_md
    dash_md = format_analytics_dashboard_markdown(sample_report)
    assert "Same-day alert" in dash_md


@pytest.mark.asyncio
async def test_tui_chat_turn_and_interactive_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERVALS_ICU_API_KEY", "test_secret")

    workout_item = {
        "date": "2026-10-27",
        "name": "Threshold Intervals",
        "sport_type": "Run",
        "description": "Main Set 4x\n- 5m 98-102% Pace\n- 2m 65% Pace",
    }

    class _FakeConfirmationGraph:
        def __init__(self) -> None:
            self.invocations: list[dict[str, Any]] = []

        def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
            self.invocations.append(state)
            msgs = list(state["messages"])
            msgs.extend(
                [
                    AIMessage(
                        content="Analyst: CTL 55, ready for threshold work.",
                        name="Performance_and_Health_Analyst",
                    ),
                    AIMessage(
                        content="Previewing workout on intervals.icu...",
                        tool_calls=[
                            {
                                "id": "call_1",
                                "name": "upload_training_plan_to_intervals",
                                "args": {"workouts": [workout_item], "confirm": False},
                            }
                        ],
                    ),
                    ToolMessage(
                        content=json.dumps(
                            {
                                "status": "confirmation_required",
                                "action": "upload_training_plan_to_intervals",
                                "summary": "Upload/update 1 workout(s) on the intervals.icu calendar.",
                                "details": {"workouts": [workout_item], "upsert": True},
                            }
                        ),
                        tool_call_id="call_1",
                        name="upload_training_plan_to_intervals",
                    ),
                    AIMessage(
                        content="Here is the threshold workout preview. Please confirm upload.",
                        name="Hybrid_Coach_Planner",
                    ),
                ]
            )
            return {
                "messages": msgs,
                "visited_agents": ["analyst", "planner"],
                "pipeline": ["analyst", "planner"],
                "router": "heuristic",
            }

    fake_graph = _FakeConfirmationGraph()
    app = KinesisTUI(graph=fake_graph, auto_refresh_on_mount=False)

    async with app.run_test(size=(120, 40)) as pilot:
        # 1. Submit a /plan command
        await app.submit_user_input("/plan threshold session tomorrow")
        await pilot.pause()

        assert len(fake_graph.invocations) == 1
        assert fake_graph.invocations[0]["mode"] == "plan"
        assert len(app.pending_confirmations) == 1

        banner = app.query_one("#confirmation-bar", ConfirmationBanner)
        assert banner.has_class("visible")

        # 2. Approve the confirmation interactively
        with respx.mock(base_url="https://intervals.icu/api/v1") as respx_mock:
            bulk_route = respx_mock.post("/athlete/0/events/bulk").mock(
                return_value=httpx.Response(
                    200,
                    json=[
                        {
                            "id": 321,
                            "name": "Threshold Intervals",
                            "type": "Run",
                            "start_date_local": "2026-10-27T00:00:00",
                        }
                    ],
                )
            )
            respx_mock.get("/athlete/0/events").mock(return_value=httpx.Response(200, json=[]))
            respx_mock.get("/athlete/0/workouts").mock(return_value=httpx.Response(200, json=[]))
            respx_mock.get("/athlete/0/folders").mock(return_value=httpx.Response(200, json=[]))
            respx_mock.get("/athlete/0/training-plan").mock(return_value=httpx.Response(200, json={}))

            await app.approve_pending_confirmations()
            await pilot.pause()

            assert bulk_route.called
            assert len(app.pending_confirmations) == 0
            assert not banner.has_class("visible")


@pytest.mark.asyncio
async def test_tui_tabs_sidebar_and_slash_commands() -> None:
    app = KinesisTUI(auto_refresh_on_mount=False)

    async with app.run_test(size=(120, 40)) as pilot:
        # Toggle sidebar
        sidebar = app.query_one("#sidebar", VerticalScroll)
        assert not sidebar.has_class("hidden")
        app.action_toggle_sidebar()
        assert sidebar.has_class("hidden")
        app.action_toggle_sidebar()
        assert not sidebar.has_class("hidden")

        # Switch tabs via actions
        tabs = app.query_one("#main-tabs", TabbedContent)
        app.action_tab_analytics()
        assert tabs.active == "tab-analytics"
        app.action_tab_calendar()
        assert tabs.active == "tab-calendar"
        app.action_tab_evidence()
        assert tabs.active == "tab-evidence"
        app.action_tab_chat()
        assert tabs.active == "tab-chat"

        # Slash commands: /status, /evidence, /clear
        await app.submit_user_input("/status")
        await app.submit_user_input("/evidence")
        cards_before_clear = len(app.query(ChatMessageCard))
        assert cards_before_clear >= 4

        await app.submit_user_input("/clear")
        cards_after_clear = len(app.query(ChatMessageCard))
        assert cards_after_clear == 1  # Welcome card restored


def test_cli_default_and_tui_subcommand_invoke_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner
    import hybrid_training_agent.tui as tui_mod

    called = {"count": 0}

    def _fake_run_tui(graph: Any = None) -> None:
        called["count"] += 1

    monkeypatch.setattr(tui_mod, "run_tui", _fake_run_tui)
    runner = CliRunner()

    res_default = runner.invoke(cli.app, [])
    assert res_default.exit_code == 0
    assert called["count"] == 1

    res_sub = runner.invoke(cli.app, ["tui"])
    assert res_sub.exit_code == 0
    assert called["count"] == 2


@pytest.mark.asyncio
async def test_tui_analytics_day_selector_and_ai_commentary() -> None:
    class _FakeAnalystGraph:
        def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
            msgs = list(state["messages"]) + [
                AIMessage(
                    content="Your 60-day CTL is trending upward with stable HRV.",
                    name="Performance_and_Health_Analyst",
                )
            ]
            return {
                "messages": msgs,
                "visited_agents": ["analyst"],
                "pipeline": ["analyst"],
                "router": "heuristic",
            }

    app = KinesisTUI(graph=_FakeAnalystGraph(), auto_refresh_on_mount=False)
    async with app.run_test(size=(110, 32)) as pilot:
        await pilot.pause()
        app.action_tab_analytics()
        await pilot.pause()

        assert app.query_one("#btn-days-28").has_class("active-window")
        app._trigger_analytics_window(60)
        await pilot.pause()
        assert app.analytics_days == 60
        assert app.query_one("#btn-days-60").has_class("active-window")
        assert not app.query_one("#btn-days-28").has_class("active-window")

        # Run AI commentary and verify it renders inside #analytics-ai-markdown
        await app._run_ai_analytics_async()
        await pilot.pause()
        ai_md = app.query_one("#analytics-ai-markdown")
        assert ai_md.has_class("visible")
        assert not app.query_one("#btn-ai-analytics").disabled

def _freeze_cursor(app: KinesisTUI) -> None:
    for input_widget in app.query(Input):
        input_widget.cursor_blink = False


def _render_ansi(app: KinesisTUI) -> str:
    width, height = app.size
    console = Console(
        width=width,
        height=height,
        file=io.StringIO(),
        force_terminal=True,
        color_system="truecolor",
        record=True,
        legacy_windows=False,
        safe_box=False,
    )
    console.print(
        app.screen._compositor.render_update(
            full=True, screen_stack=app._background_screens, simplify=False
        )
    )
    return console.export_text(styles=True)


def _assert_snapshot(name: str, app: KinesisTUI) -> None:
    snapshot = SNAPSHOT_DIR / name
    actual = _render_ansi(app).replace("\r\n", "\n")
    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(actual, encoding="utf-8")
        return
    assert snapshot.exists(), f"Missing golden frame {snapshot}"
    expected = snapshot.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert actual == expected


class _SnapshotConfirmationGraph:
    """Deterministic graph that previews a mutating upload."""

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        msgs = list(state["messages"])
        msgs.extend(
            [
                AIMessage(
                    content="CTL 13.8, ATL 19.9, TSB -6.1. HRV data is missing for the window.",
                    name="Performance_and_Health_Analyst",
                ),
                AIMessage(
                    content="Previewing the week on intervals.icu...",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "upload_training_plan_to_intervals",
                            "args": {"workouts": [], "confirm": False},
                        }
                    ],
                ),
                ToolMessage(
                    content=json.dumps(
                        {
                            "status": "confirmation_required",
                            "action": "upload_training_plan_to_intervals",
                            "summary": "Upload/update 3 workout(s) on the intervals.icu calendar (upsert=True).",
                            "details": {"workouts": [], "upsert": True},
                        }
                    ),
                    tool_call_id="call_1",
                    name="upload_training_plan_to_intervals",
                ),
                AIMessage(
                    content="### Week of 26 Oct — Hybrid Block (Build)\n\n**Load check:** CTL 13.8.",
                    name="Hybrid_Coach_Planner",
                ),
            ]
        )
        return {
            "messages": msgs,
            "visited_agents": ["analyst", "planner"],
            "pipeline": ["analyst", "planner"],
            "router": "heuristic",
        }


def test_markdown_formatters_never_leak_none_and_show_verdicts() -> None:
    report = {
        "analysis_window_days": 28,
        "load_and_form": {
            "ctl_fitness": 52.0,
            "atl_fatigue": 48.0,
            "tsb_form": -6.1,
            "acwr_ratio": 1.7,
            "acwr_interpretation": None,
        },
        "recovery_and_wellness": {
            "hrv": {},
            "resting_hr": {"latest_rhr_bpm": None},
            "sleep_and_subjective": {},
        },
        "running_summary": {
            "weekly_avg_distance_km": 30.0,
            "sessions_per_week": 3.0,
            "intensity_distribution_3_zone": {},
        },
        "strength_summary": {},
        "concurrent_interference_audit": {"warnings_count": 3, "warnings": []},
    }
    sidebar_md = format_readiness_sidebar_markdown(report, as_of="12:34")
    assert "None" not in sidebar_md
    assert "as of 12:34" in sidebar_md
    assert "✓ steady" in sidebar_md  # TSB verdict
    assert "✗ high" in sidebar_md  # ACWR verdict
    assert "✗ act" in sidebar_md  # alert verdict
    assert "no data" in sidebar_md
    assert "No HRV data in this window" in sidebar_md
    # One metric per row, not a collapsed paragraph.
    metric_lines = [
        line
        for line in sidebar_md.splitlines()
        if line.startswith(("CTL", "ATL", "TSB", "ACWR", "HRV", "RHR", "Sleep", "Run", "Zones", "Str", "Alerts"))
    ]
    assert len(metric_lines) == 11
    assert all("  " in line or "RHR" in line for line in metric_lines)

    dash_md = format_analytics_dashboard_markdown(report, as_of="12:34")
    assert "None" not in dash_md
    assert "as of 12:34" in dash_md
    assert "✗ high" in dash_md


def test_library_and_evidence_use_human_titles_and_counts() -> None:
    workouts = [
        {"name": f"Workout {i}", "type": "Run", "moving_time": 2400}
        for i in range(171)
    ]
    library_md = format_library_markdown(
        {
            "folders": [
                {
                    "id": 783382,
                    "name": "Endurance Base",
                    "num_workouts": 12,
                }
            ],
            "workouts": workouts,
        }
    )
    assert "1 folder · 171 workouts" in library_md
    assert "showing 15 of 171" in library_md
    assert "**Endurance Base** — 12 workouts · `#783382`" in library_md
    assert "- **Workout 0**" in library_md
    assert "Workout 16" not in library_md

    curated = {
        "concurrent_interference": [
            {
                "title": "Concurrent training and interference",
                "authors": "Smith J",
                "year": 2021,
                "journal": "Sports Medicine",
                "pmid": "12345",
                "practical_application": "Separate hard runs and heavy lifting by 6-24 h.",
            }
        ]
    }
    evidence_md = format_evidence_markdown(curated=curated)
    assert "Concurrent Training Interference" in evidence_md
    assert "concurrent_interference" not in evidence_md
    assert "**Concurrent training and interference**" in evidence_md
    assert "Smith J · 2021 · *Sports Medicine*" in evidence_md
    assert "PMID `12345`" in evidence_md
    assert "Application:" in evidence_md

    live_md = format_evidence_markdown(
        papers=[
            {
                "title": "A paper",
                "authors": None,
                "year": None,
                "journal": None,
                "pmid": None,
                "doi": None,
                "abstract": "Short abstract.",
            }
        ],
        query="test",
    )
    assert "None" not in live_md
    assert "n.d." in live_md


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("size", "name"),
    [
        ((120, 40), "chat-normal-120x40.ansi"),
        ((80, 24), "chat-normal-80x24.ansi"),
        ((60, 24), "chat-normal-60x24.ansi"),
    ],
)
async def test_tui_golden_frames(size: tuple[int, int], name: str) -> None:
    app = KinesisTUI(auto_refresh_on_mount=False)
    async with app.run_test(size=size) as pilot:
        await pilot.pause(0.25)
        _freeze_cursor(app)
        await pilot.pause(0.1)
        _assert_snapshot(name, app)


@pytest.mark.asyncio
async def test_tui_golden_confirmation_frame() -> None:
    app = KinesisTUI(graph=_SnapshotConfirmationGraph(), auto_refresh_on_mount=False)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(0.25)
        await app.submit_user_input("/plan build week from Monday")
        await pilot.pause(0.4)
        _freeze_cursor(app)
        await pilot.pause(0.1)
        _assert_snapshot("chat-confirm-80x24.ansi", app)


@pytest.mark.asyncio
async def test_tui_golden_too_small_frame() -> None:
    app = KinesisTUI(auto_refresh_on_mount=False)
    async with app.run_test(size=(60, 18)) as pilot:
        await pilot.pause(0.35)
        _freeze_cursor(app)
        _assert_snapshot("too-small-60x18.ansi", app)


@pytest.mark.asyncio
async def test_tui_layout_breakpoints() -> None:
    app = KinesisTUI(auto_refresh_on_mount=False)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        sidebar = app.query_one("#sidebar", VerticalScroll)
        assert not sidebar.has_class("hidden")
        assert not app.query_one("#analytics-activities-pane").has_class("pane-hidden")
        assert not app.query_one("#library-pane").has_class("pane-hidden")
        bottom = app.query_one("#bottom-bar")
        assert bottom.region.height == 1
        assert bottom.region.y == 39
        tabs = app.query_one("#main-tabs", TabbedContent)
        assert tabs.active == "tab-chat"

    app = KinesisTUI(auto_refresh_on_mount=False)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(0.2)
        sidebar = app.query_one("#sidebar", VerticalScroll)
        assert sidebar.has_class("hidden")
        # ctrl+b restores the sidebar even below the auto-hide breakpoint.
        app.action_toggle_sidebar()
        assert not sidebar.has_class("hidden")
        app.action_toggle_sidebar()
        assert sidebar.has_class("hidden")
        assert app.query_one("#analytics-activities-pane").has_class("pane-hidden")
        assert app.query_one("#library-pane").has_class("pane-hidden")
        chat_input = app.query_one("#chat-input", Input)
        assert chat_input.display
        assert chat_input.region.bottom <= 24
        assert app.query_one("#chat-main").region.width == 80
        hints = app.query_one("#bottom-hints")
        keys = app.query_one("#bottom-keys")
        assert "1-4 tabs" in str(hints.content)
        assert hints.region.right <= keys.region.x or keys.region.right <= hints.region.x

    app = KinesisTUI(auto_refresh_on_mount=False)
    async with app.run_test(size=(60, 24)) as pilot:
        await pilot.pause(0.2)
        assert app.query_one("#sidebar", VerticalScroll).has_class("hidden")
        assert app.query_one("#chat-main").region.width == 60
        assert app.query_one("#chat-input", Input).region.bottom <= 24

    app = KinesisTUI(auto_refresh_on_mount=False)
    async with app.run_test(size=(60, 18)) as pilot:
        await pilot.pause(0.35)
        assert app._is_too_small
        overlay = app.query_one("#too-small", StatusOverlay)
        assert overlay.display
        assert "current 60×18 · needs 80×24 · q to quit" in str(overlay.content)
        assert not app.query_one("#top-bar").display
        assert not app.query_one("#bottom-bar").display
        assert isinstance(app.focused, StatusOverlay)
        await pilot.press("q")
        await pilot.pause(0.2)
        assert not app.is_running


@pytest.mark.asyncio
async def test_confirmation_banner_layout_and_keys() -> None:
    app = KinesisTUI(graph=_SnapshotConfirmationGraph(), auto_refresh_on_mount=False)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(0.2)
        await app.submit_user_input("/plan build week from Monday")
        await pilot.pause(0.3)

        banner = app.query_one("#confirmation-bar", ConfirmationBanner)
        summary = app.query_one("#confirmation-summary")
        actions = app.query_one("#confirmation-actions")
        chat_input = app.query_one("#chat-input", Input)

        assert banner.display and banner.has_class("visible")
        assert isinstance(app.focused, ConfirmationBanner)
        assert summary.region.height == 2
        assert len(str(summary.content).splitlines()) <= 2
        assert summary.region.y + summary.region.height <= actions.region.y
        assert actions.region.height == 1
        assert chat_input.display
        assert chat_input.region.bottom <= 24
        assert app.query_one("#chat-scroll", VerticalScroll).region.height >= 5

        await pilot.press("r")
        await pilot.pause(0.3)
        assert app.pending_confirmations == []
        assert not banner.has_class("visible")
        assert isinstance(app.focused, Input)


@pytest.mark.asyncio
async def test_busy_input_queues_and_escape_stops() -> None:
    class _SlowGraph:
        def __init__(self) -> None:
            self.gate = threading.Event()

        def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
            self.gate.wait(5)
            return {"messages": list(state["messages"])}

    graph = _SlowGraph()
    app = KinesisTUI(graph=graph, auto_refresh_on_mount=False)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(0.2)
        task = asyncio.create_task(app.submit_user_input("first"))
        await pilot.pause(0.2)
        assert app._is_busy

        chat_input = app.query_one("#chat-input", Input)
        chat_input.value = "second"
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert app._queued_messages == ["second"]
        assert chat_input.value == ""
        status = str(app.query_one("#agent-activity-bar").content)
        assert "Thinking" in status and "esc stop" in status

        app._queued_messages.clear()
        await pilot.press("escape")
        await pilot.pause(0.3)
        assert not app._is_busy
        assert "Stopped." in str(app.query_one("#agent-activity-bar").content)
        graph.gate.set()
        await task


@pytest.mark.asyncio
async def test_mode_chip_shows_and_resets_after_one_turn() -> None:
    class _OneTurnGraph:
        def __init__(self) -> None:
            self.modes: list[str] = []

        def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
            self.modes.append(state["mode"])
            msgs = list(state["messages"]) + [
                AIMessage(content="Done.", name="Hybrid_Coach_Planner")
            ]
            return {
                "messages": msgs,
                "visited_agents": ["planner"],
                "pipeline": ["planner"],
                "router": "heuristic",
            }

    graph = _OneTurnGraph()
    app = KinesisTUI(graph=graph, auto_refresh_on_mount=False)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        app.action_quick_plan()
        await pilot.pause(0.1)
        chip = app.query_one("#mode-chip")
        assert str(chip.content) == "plan"
        assert chip.has_class("plan")

        await app.submit_user_input("5x1km at threshold")
        await pilot.pause(0.2)
        assert graph.modes == ["plan"]
        assert str(chip.content) == "auto"
        assert not chip.has_class("plan")


def _contrast_ratio(fg: str, bg: str) -> float:
    def luminance(hex_color: str) -> float:
        value = hex_color.lstrip("#")
        red, green, blue = (int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))

        def channel(component: float) -> float:
            return (
                component / 12.92
                if component <= 0.03928
                else ((component + 0.055) / 1.055) ** 2.4
            )

        return (
            0.2126 * channel(red)
            + 0.7152 * channel(green)
            + 0.0722 * channel(blue)
        )

    light, dark = sorted((luminance(fg), luminance(bg)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def test_palette_contrast_floors() -> None:
    from hybrid_training_agent.tui import _PALETTE

    assert _contrast_ratio(_PALETTE["muted"], _PALETTE["panel"]) >= 4.5
    assert _contrast_ratio(_PALETTE["muted"], _PALETTE["base"]) >= 4.5
    assert _contrast_ratio(_PALETTE["accent"], _PALETTE["surface"]) >= 4.5
    assert _contrast_ratio(_PALETTE["base"], _PALETTE["accent"]) >= 4.5
    assert _contrast_ratio(_PALETTE["line"], _PALETTE["surface"]) >= 3.0
    assert _contrast_ratio(_PALETTE["focus"], _PALETTE["surface"]) >= 3.0
    assert _contrast_ratio(_PALETTE["status"], _PALETTE["panel"]) >= 4.5
    assert _contrast_ratio(_PALETTE["danger"], _PALETTE["surface"]) >= 4.5


@pytest.mark.asyncio
async def test_table_cells_ellipsize_and_calendar_has_no_id_column() -> None:
    from hybrid_training_agent.tui import _short_date, _table_cell

    assert _short_date("2026-10-25T07:00:00") == "10-25"
    clipped = _table_cell("Full-Body Strength Endurance", width=13)
    assert clipped.no_wrap
    assert clipped.overflow == "ellipsis"
    right = str(_table_cell(61, width=5, right=True).plain)
    assert right == "   61"

    app = KinesisTUI(auto_refresh_on_mount=False)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        calendar = app.query_one("#calendar-table", DataTable)
        labels = [str(column.label) for column in calendar.ordered_columns]
        assert labels == ["Date", "Type", "Name", "Min", "Load"]
        assert "ID" not in labels
        activities = app.query_one("#activities-table", DataTable)
        activity_labels = [str(column.label) for column in activities.ordered_columns]
        assert activity_labels == ["Date", "Sport", "Name", "km", "Min", "Load"]


@pytest.mark.asyncio
async def test_number_and_ctrl_number_tab_keys() -> None:
    app = KinesisTUI(auto_refresh_on_mount=False)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        tabs = app.query_one("#main-tabs", TabbedContent)
        await pilot.press("ctrl+2")
        await pilot.pause(0.2)
        assert tabs.active == "tab-analytics"
        await pilot.press("ctrl+3")
        await pilot.pause(0.2)
        assert tabs.active == "tab-calendar"
        await pilot.press("ctrl+4")
        await pilot.pause(0.2)
        assert tabs.active == "tab-evidence"
        await pilot.press("ctrl+1")
        await pilot.pause(0.2)
        assert tabs.active == "tab-chat"

        # Bare numbers work whenever the chat input is not capturing text.
        app.query_one("#chat-scroll").focus()
        await pilot.press("2")
        await pilot.pause(0.2)
        assert tabs.active == "tab-analytics"


@pytest.mark.asyncio
async def test_chrome_has_one_row_top_tabs_footer_and_no_duplicate_slash_hints() -> None:
    app = KinesisTUI(auto_refresh_on_mount=False)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(0.2)
        assert app.query_one("#top-bar").region.height == 1
        assert app.query_one("#bottom-bar").region.height == 1
        assert "⚡" not in str(app.query_one("#top-brand").content)
        tabs = app.query_one("#main-tabs", TabbedContent).query_one("Tabs")
        assert tabs.region.height == 2
        assert "/plan" not in str(app.query_one("#top-telemetry").content)
        assert "/plan" not in app.query_one("#chat-input", Input).placeholder
        assert "/plan" not in str(app.query_one("#bottom-hints").content)
        assert "/plan" not in str(app.query_one("#bottom-keys").content)

        app.action_tab_analytics()
        await pilot.pause(0.1)
        assert app.query_one("#analytics-toolbar").region.height == 2

        app.action_tab_evidence()
        await pilot.pause(0.1)
        assert app.query_one("#evidence-toolbar").region.height == 2
        evidence_controls = [
            app.query_one("#evidence-topic-select"),
            app.query_one("#evidence-query-input"),
            app.query_one("#btn-search-evidence"),
            app.query_one("#btn-curated-evidence"),
        ]
        assert {control.region.y for control in evidence_controls} == {3}
        assert all(control.region.height == 1 for control in evidence_controls)

@pytest.mark.asyncio
async def test_confirmation_summary_ellipsizes_at_narrow_width() -> None:
    app = KinesisTUI(auto_refresh_on_mount=False)
    async with app.run_test(size=(60, 24)) as pilot:
        await pilot.pause(0.2)
        app.pending_confirmations = [
            PendingConfirmation(
                tool_name="upload_training_plan_to_intervals",
                tool_args={},
                action="upload_training_plan_to_intervals",
                summary="Upload/update 12 workout(s) on the intervals.icu calendar "
                "and replace matching events for the whole block. " * 4,
                details={},
            )
        ]
        app._update_confirmation_banner()
        await pilot.pause(0.25)
        summary = app.query_one("#confirmation-summary")
        text = str(summary.content)
        assert summary.region.height == 2
        assert len(text.splitlines()) <= 2
        assert text.rstrip().endswith("…")
        assert app.query_one("#chat-input", Input).display
        assert app.query_one("#chat-input", Input).region.bottom <= 24
