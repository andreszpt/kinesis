from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from textual.containers import VerticalScroll
from textual.widgets import TabbedContent

from hybrid_training_agent import cli
from hybrid_training_agent.clients.http_client import reset_shared_http_client
from hybrid_training_agent.clients.intervals_client import _intervals_cache
from hybrid_training_agent.clients.science_client import _science_cache
from hybrid_training_agent.tui import (
    ChatMessageCard,
    ConfirmationBanner,
    KinesisTUI,
    PendingConfirmation,
    execute_pending_confirmation,
    extract_pending_confirmations,
    format_analytics_dashboard_markdown,
    format_calendar_sidebar_markdown,
    format_confirmation_result_markdown,
    format_evidence_markdown,
    format_readiness_sidebar_markdown,
)


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
