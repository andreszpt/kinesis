from __future__ import annotations

import io
from typing import Any

import httpx
import pytest
import respx
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from hybrid_training_agent.agents import graph as graph_module
from hybrid_training_agent.agents.graph import (
    _detect_days_back,
    _select_analyst_tools,
    _select_planner_tools,
    build_hybrid_training_graph,
    classify_intent,
    classify_intent_pipeline,
)
from hybrid_training_agent import cli
from hybrid_training_agent.cli import _pending_question, extract_message_text
from hybrid_training_agent.clients.cache import TTLCache
from hybrid_training_agent.clients.http_client import reset_shared_http_client
from hybrid_training_agent.clients.intervals_client import (
    IntervalsIcuClient,
    _intervals_cache,
    validate_intervals_run_syntax,
)
from hybrid_training_agent.clients.science_client import (
    ScienceLiteratureClient,
    _science_cache,
)
from hybrid_training_agent.config import Settings, get_chat_model
from hybrid_training_agent.models import PlannedWorkoutInput
from hybrid_training_agent.tools.analytics_tools import compute_hybrid_analytics
from hybrid_training_agent.tools.planner_tools import audit_planned_workouts


class _RecordingFakeChatModel(GenericFakeChatModel):
    """Scripted fake chat model that plays nicely with `bind_tools`."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


@pytest.fixture(autouse=True)
def _hermetic_tooling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests offline and isolated from the shared read caches."""
    monkeypatch.setenv("INTERVALS_ICU_API_KEY", "")
    reset_shared_http_client()
    _intervals_cache.clear()
    _science_cache.clear()
    yield
    _intervals_cache.clear()
    _science_cache.clear()
    reset_shared_http_client()


def test_validate_intervals_run_syntax_valid_and_invalid() -> None:
    valid_desc = (
        "Warmup\n"
        "- 15m 65-75% Pace\n\n"
        "Main Set 5x\n"
        "- 1km 98-102% Pace\n"
        "- 2m 60-70% Pace\n\n"
        "Cooldown\n"
        "- 10m 65-75% Pace"
    )
    res = validate_intervals_run_syntax(valid_desc)
    assert res["is_valid_structured_workout"] is True
    assert res["valid_duration_steps"] == 4
    assert res["steps_with_intensity_targets"] == 4
    assert res["issues"] == []

    invalid_desc = "Go for a 45 minute easy run around the park."
    bad_res = validate_intervals_run_syntax(invalid_desc)
    assert bad_res["is_valid_structured_workout"] is False
    assert len(bad_res["issues"]) > 0


def test_pubmed_xml_parsing_and_curated_evidence() -> None:
    sample_xml = """<?xml version="1.0" ?>
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>34757594</PMID>
          <Article>
            <Journal>
              <Title>Sports Medicine</Title>
              <JournalIssue>
                <PubDate><Year>2022</Year></PubDate>
              </JournalIssue>
            </Journal>
            <ArticleTitle>Compatibility of Concurrent Aerobic and Strength Training</ArticleTitle>
            <Abstract>
              <AbstractText Label="RESULTS">Concurrent training did not attenuate maximal strength.</AbstractText>
            </Abstract>
            <AuthorList>
              <Author><LastName>Schumann</LastName><Initials>M</Initials></Author>
              <Author><LastName>Lundberg</LastName><Initials>TR</Initials></Author>
            </AuthorList>
            <PublicationTypeList>
              <PublicationType>Meta-Analysis</PublicationType>
            </PublicationTypeList>
          </Article>
        </MedlineCitation>
        <PubmedData>
          <ArticleIdList>
            <ArticleId IdType="doi">10.1007/s40279-021-01587-7</ArticleId>
          </ArticleIdList>
        </PubmedData>
      </PubmedArticle>
    </PubmedArticleSet>
    """
    parsed = ScienceLiteratureClient._parse_pubmed_xml(sample_xml)
    assert len(parsed) == 1
    assert parsed[0]["pmid"] == "34757594"
    assert parsed[0]["doi"] == "10.1007/s40279-021-01587-7"
    assert "Schumann M" in parsed[0]["authors"]
    assert "Meta-Analysis" in parsed[0]["publication_types"]

    curated = ScienceLiteratureClient.get_curated_evidence("concurrent_interference")
    assert "concurrent_interference" in curated
    assert any(c["pmid"] == "34757594" for c in curated["concurrent_interference"])


def test_compute_hybrid_analytics_and_interference() -> None:
    wellness = [
        {"id": "2026-10-11", "ctl": 50.0, "atl": 55.0, "hrv": 68.0, "restingHR": 48, "sleepSecs": 28800},
        {"id": "2026-10-12", "ctl": 51.0, "atl": 56.0, "hrv": 70.0, "restingHR": 47, "sleepSecs": 27000},
        {"id": "2026-10-13", "ctl": 52.0, "atl": 58.0, "hrv": 66.0, "restingHR": 49, "sleepSecs": 28000},
        {"id": "2026-10-14", "ctl": 53.0, "atl": 60.0, "hrv": 69.0, "restingHR": 48, "sleepSecs": 29000},
    ]
    activities = [
        {
            "id": "run1",
            "type": "Run",
            "name": "Easy Aerobic Run",
            "start_date_local": "2026-10-13T08:00:00",
            "distance": 10000.0,
            "moving_time": 3000,
            "average_speed": 3.33,
            "average_heartrate": 140,
            "icu_training_load": 58,
            "icu_pace_zone_times": [1200, 1500, 200, 100, 0, 0, 0],
        },
        {
            "id": "suunto_str1",
            "type": "WeightTraining",
            "name": "Strength Training",
            "start_date_local": "2026-10-13T11:00:00",  # Only 3h after the run!
            "moving_time": 3300,
            "average_heartrate": 118,
            "icu_training_load": 25,
        },
    ]

    report = compute_hybrid_analytics(
        wellness_records=wellness,
        activities=activities,
        days_back=7,
    )

    assert report["load_and_form"]["ctl_fitness"] == 53.0
    assert report["load_and_form"]["atl_fatigue"] == 60.0
    assert report["load_and_form"]["tsb_form"] == -7.0
    assert report["load_and_form"]["acwr_ratio"] == round(60.0 / 53.0, 2)
    assert report["running_summary"]["intensity_distribution_3_zone"]["zone_1_2_low_pct"] == 90.0
    assert report["strength_summary"]["total_sessions"] == 1
    assert report["concurrent_interference_audit"]["warnings_count"] == 1
    assert "3.0h apart" in report["concurrent_interference_audit"]["warnings"][0]


def test_audit_planned_workouts_and_bulk_upload() -> None:
    workouts = [
        PlannedWorkoutInput(
            date="2026-10-19",
            name="Heavy Neural Lower Body Strength",
            sport_type="WeightTraining",
            description="Back Squat 4x5, RDL 3x6, Calf Raises 3x10",
            moving_time_minutes=50,
            estimated_load=55,
        ),
        PlannedWorkoutInput(
            date="2026-10-20",
            name="VO2max Track Intervals",
            sport_type="Run",
            description="Warmup\n- 15m 65-75% Pace\n\nMain Set 5x\n- 1km 102-106% Pace\n- 2m 60-70% Pace\n\nCooldown\n- 10m 65-75% Pace",
            moving_time_minutes=50,
            estimated_load=72,
        ),
    ]
    audit = audit_planned_workouts(workouts)
    assert audit["all_workouts_valid"] is True
    assert len(audit["interference_audit_warnings"]) == 1
    assert "Sequential Interference Caution" in audit["interference_audit_warnings"][0]

    settings = Settings(intervals_api_key="test_secret", intervals_athlete_id="0")
    client = IntervalsIcuClient(settings=settings)
    with respx.mock(base_url="https://intervals.icu/api/v1") as respx_mock:
        post_route = respx_mock.post("/athlete/0/events/bulk").mock(
            return_value=httpx.Response(200, json=[{"id": 101}, {"id": 102}])
        )
        uploaded = client.create_events_bulk(workouts, upsert=True)
        assert post_route.called
        assert len(uploaded) == 2

        import json

        sent = json.loads(post_route.calls[0].request.content)
        # Strength sessions are uploaded as calendar notes so intervals.icu never attempts a
        # Suunto guide push (which fails for a step-less strength workout).
        assert sent[0]["category"] == "NOTE"
        assert "type" not in sent[0]
        assert "icu_training_load" not in sent[0]
        assert sent[0]["description"] == "Back Squat 4x5, RDL 3x6, Calf Raises 3x10"
        # Running workouts remain structured WORKOUT events with a load for Suunto sync.
        assert sent[1]["category"] == "WORKOUT"
        assert sent[1]["type"] == "Run"
        assert sent[1]["icu_training_load"] == 72


def test_strength_events_are_uploaded_as_notes() -> None:
    strength = PlannedWorkoutInput(
        date="2026-10-18",
        name="Full-Body Strength",
        sport_type="WeightTraining",
        description="Squat 3x5",
        moving_time_minutes=60,
        estimated_load=35,
    )
    payload = IntervalsIcuClient.build_event_payload(strength)

    assert payload["category"] == "NOTE"
    assert "type" not in payload
    assert "icu_training_load" not in payload
    assert "moving_time" not in payload
    assert payload["start_date_local"] == "2026-10-18T00:00:00"
    assert payload["description"] == "Squat 3x5"
    assert payload["external_id"].startswith("kinesis-2026-10-18-weighttraining-")

    run = PlannedWorkoutInput(
        date="2026-10-19",
        name="Easy Run",
        sport_type="Run",
        description="- 30m 70% Pace",
        moving_time_minutes=30,
        estimated_load=30,
    )
    run_payload = IntervalsIcuClient.build_event_payload(run)
    assert run_payload["category"] == "WORKOUT"
    assert run_payload["type"] == "Run"
    assert run_payload["icu_training_load"] == 30

    note_payload = IntervalsIcuClient.build_event_payload(
        PlannedWorkoutInput(
            date="2026-10-20", name="Travel day", sport_type="Note", description="Rest"
        )
    )
    assert note_payload["category"] == "NOTE"


def test_langgraph_multi_agent_pipeline_execution() -> None:
    # Routine plan requests skip the literature sub-agent by default.
    assert classify_intent_pipeline("Build me a 7-day hybrid running and strength plan") == [
        "analyst",
        "planner",
    ]
    # ...but an explicit ask for evidence/citations brings the researcher back in.
    assert classify_intent_pipeline(
        "Build me a 7-day hybrid running and strength plan backed by scientific evidence"
    ) == ["analyst", "researcher", "planner"]
    assert classify_intent_pipeline("What does PubMed say about the interference effect?") == [
        "researcher"
    ]
    assert classify_intent_pipeline("Analyze my HRV and CTL from intervals.icu") == [
        "analyst"
    ]
    # Data questions about the week must not trigger the planner.
    assert classify_intent_pipeline("how many kms did I run this week") == ["analyst"]
    assert classify_intent_pipeline("how many workouts did I do this week") == ["analyst"]
    assert classify_intent_pipeline("what's my training plan?") == ["analyst"]
    # ...but explicit creation still plans.
    assert classify_intent_pipeline(
        "create a 60m full body strength training for tomorrow"
    ) == ["analyst", "planner"]

    class ToolEnabledFakeChatModel(GenericFakeChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self

    fake_llm = ToolEnabledFakeChatModel(
        messages=iter(
            [
                AIMessage(content="Analyst Report: CTL is 54, ATL is 50, TSB is +4, HRV is stable."),
                AIMessage(
                    content="Researcher Report: Schumann et al. (2022, PMID: 34757594) and Blagrove et al. (2018, PMID: 29249083) support 2 heavy strength sessions/wk spaced >6-24h from key runs."
                ),
                AIMessage(
                    content="Coach Plan: Here is your 7-day polarized running + neural strength plan formatted for intervals.icu and Suunto."
                ),
            ]
        )
    )

    graph = build_hybrid_training_graph(llm=fake_llm)
    state_out: dict[str, Any] = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Create a comprehensive weekly hybrid training plan for me, "
                        "backed by peer-reviewed evidence"
                    )
                )
            ],
            "next_agent": "supervisor",
            "visited_agents": [],
            "mode": "plan",
        }
    )

    assert state_out["visited_agents"] == ["analyst", "researcher", "planner"]
    ai_messages = [m for m in state_out["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) == 3
    assert ai_messages[0].name == "Performance_and_Health_Analyst"
    assert ai_messages[1].name == "Sports_Science_Researcher"
    assert ai_messages[2].name == "Hybrid_Coach_Planner"


def test_default_plan_skips_researcher_subagent() -> None:
    fake_llm = _RecordingFakeChatModel(
        messages=iter(
            [
                AIMessage(content="Analyst Report: CTL 54."),
                AIMessage(content="Coach Plan: 60m full body strength session."),
            ]
        )
    )
    graph = build_hybrid_training_graph(llm=fake_llm)
    state_out: dict[str, Any] = graph.invoke(
        {
            "messages": [
                HumanMessage(content="Create a 60m full body strength training for tomorrow")
            ],
            "next_agent": "supervisor",
            "visited_agents": [],
            "mode": "auto",
        }
    )

    assert state_out["visited_agents"] == ["analyst", "planner"]
    ai_messages = [m for m in state_out["messages"] if isinstance(m, AIMessage)]
    assert [m.name for m in ai_messages] == [
        "Performance_and_Health_Analyst",
        "Hybrid_Coach_Planner",
    ]
    assert isinstance(state_out["messages"][-1].content, str)


def test_subagent_requests_never_replay_foreign_thinking_blocks() -> None:
    """Anthropic rejects thinking signatures created under another conversation's system/tools."""
    seen_conversations: list[list[AnyMessage]] = []

    class RecordingFakeChatModel(GenericFakeChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self

        def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
            seen_conversations.append(list(input))
            return super().invoke(input, config=config, **kwargs)

    poisoned_analyst_report = AIMessage(
        content=[
            {"type": "thinking", "thinking": "secret reasoning", "signature": "sig-from-analyst"},
            {"type": "text", "text": "Analyst Report: CTL 54, ATL 50, TSB +4."},
        ]
    )

    fake_llm = RecordingFakeChatModel(
        messages=iter(
            [
                poisoned_analyst_report,
                AIMessage(content="Researcher Report: ok."),
                AIMessage(content="Coach Plan: ok."),
            ]
        )
    )
    graph = build_hybrid_training_graph(llm=fake_llm)
    graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Create a comprehensive hybrid training plan with scientific citations"
                )
            ],
            "next_agent": "supervisor",
            "visited_agents": [],
            "mode": "plan",
        }
    )

    assert len(seen_conversations) == 3
    # The researcher's request is the first one that replays the analyst transcript.
    researcher_conversation = seen_conversations[1]
    assert not isinstance(researcher_conversation[-1], AIMessage)
    for message in researcher_conversation:
        blocks = message.content if isinstance(message.content, list) else []
        assert not any(
            isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking")
            for b in blocks
        )
        assert "sig-from-analyst" not in str(message.content)

    # The briefing keeps the analyst's plain-text conclusion, tagged with its agent name.
    assert any(
        isinstance(m, AIMessage) and "Analyst Report: CTL 54" in str(m.content)
        for m in researcher_conversation
    )
    assert any(
        isinstance(m, AIMessage) and "Performance_and_Health_Analyst" in str(m.content)
        for m in researcher_conversation
    )


def test_pending_question_detection() -> None:
    assert _pending_question([HumanMessage(content="hi")]) == ""
    assert _pending_question(
        [HumanMessage(content="hi"), AIMessage(content="Here is your plan.")]
    ) == ""
    assert _pending_question(
        [
            HumanMessage(content="hi"),
            AIMessage(
                content=[
                    {"type": "thinking", "thinking": "should I ask?", "signature": "sig"},
                    {"type": "text", "text": "Do you want me to upload this to intervals.icu?"},
                ],
                name="Hybrid_Coach_Planner",
            ),
            HumanMessage(content="yes"),
            AIMessage(content="Done."),
        ]
    ) == ""


def test_ask_cmd_offers_follow_up_when_agent_asks_a_question(monkeypatch: pytest.MonkeyPatch) -> None:
    class TtyStringIO(io.StringIO):
        def isatty(self) -> bool:
            return True

    fake_llm = _RecordingFakeChatModel(
        messages=iter(
            [
                AIMessage(content="Should the session be upper or lower body focused?"),
                AIMessage(content="Planner: here is the 60m full body session."),
            ]
        )
    )
    monkeypatch.setattr(cli.sys, "stdin", TtyStringIO())
    monkeypatch.setattr(cli, "build_hybrid_training_graph", lambda: build_hybrid_training_graph(llm=fake_llm))

    replies = iter(["Lower body focus with a full-body finisher."])
    monkeypatch.setattr(cli.Prompt, "ask", lambda *a, **k: next(replies))

    cli.ask_cmd(prompt="Create a 60m full body strength session", mode="research")

    # Both fake replies were consumed: the follow-up answer triggered a second graph turn.
    assert list(fake_llm.messages) == []


def test_ask_cmd_is_silent_when_not_a_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_llm = _RecordingFakeChatModel(
        messages=iter([AIMessage(content="Should the session be upper or lower body focused?")])
    )
    monkeypatch.setattr(cli, "build_hybrid_training_graph", lambda: build_hybrid_training_graph(llm=fake_llm))
    monkeypatch.setattr(cli.Prompt, "ask", lambda *a, **k: pytest.fail("must not prompt without a TTY"))

    cli.ask_cmd(prompt="Create a 60m full body strength session", mode="research")

    # The initial run consumed the scripted reply and no follow-up turn ran.
    assert len(list(fake_llm.messages)) == 0


def test_subagent_scope_boundaries_keep_uploads_in_the_planner() -> None:
    from hybrid_training_agent.agents.prompts import (
        ANALYST_SYSTEM_PROMPT,
        PLANNER_SYSTEM_PROMPT,
        RESEARCHER_SYSTEM_PROMPT,
    )

    for prompt in (ANALYST_SYSTEM_PROMPT, RESEARCHER_SYSTEM_PROMPT):
        lower = prompt.lower()
        assert "scope & boundaries" in lower
        assert "never mention uploads" in lower
        assert "planner" in lower

    # The planner remains the only sub-agent wired to actually upload.
    assert "upload_training_plan_to_intervals" in PLANNER_SYSTEM_PROMPT


def test_planner_makes_warmup_and_cooldown_opt_in() -> None:
    from hybrid_training_agent.agents.prompts import PLANNER_SYSTEM_PROMPT
    from hybrid_training_agent.models import PlannedWorkoutInput

    lower = PLANNER_SYSTEM_PROMPT.lower()
    assert "warm-up & cool-down are opt-in" in lower
    assert "do not add warm-up or cool-down" in lower
    # The running example must not demonstrate a default warm-up/cool-down block.
    assert "Warmup\n- 15m" not in PLANNER_SYSTEM_PROMPT
    assert "Cooldown\n- 10m" not in PLANNER_SYSTEM_PROMPT

    field_desc = PlannedWorkoutInput.model_fields["description"].description or ""
    assert "Do NOT add warm-up or cool-down steps" in field_desc
    assert "Do not add warm-up or cool-down sections" in field_desc


def test_opencode_go_provider_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test_go_key")
    monkeypatch.setenv("LLM_PROVIDER", "opencode-go")

    # 1. Chat completions model (e.g. kimi-k3)
    m_chat = get_chat_model(provider="opencode-go", model="opencode-go/kimi-k3")
    assert getattr(m_chat, "model_name", None) == "kimi-k3"
    assert getattr(m_chat, "openai_api_base", None) == "https://opencode.ai/inference/go/openai/v1"
    assert not getattr(m_chat, "use_responses_api", False)

    # 2. Responses API model (e.g. gpt-6-luna)
    m_resp = get_chat_model(provider="opencode-go", model="gpt-6-luna")
    assert getattr(m_resp, "model_name", None) == "gpt-6-luna"
    assert getattr(m_resp, "use_responses_api", False) is True

    # 3. Anthropic Messages API model (e.g. qwen3.8-max)
    m_anth = get_chat_model(provider="opencode-go", model="qwen3.8-max")
    assert getattr(m_anth, "model", None) == "qwen3.8-max"
    assert getattr(m_anth, "anthropic_api_url", None) == "https://opencode.ai/inference/go/anthropic"


def test_deep_analysis_and_library_tool_selection() -> None:
    assert _select_analyst_tools("create a 60m full body strength session") == []
    deep = {t.name for t in _select_analyst_tools(
        "analyze how I executed my last VO2max session and my best efforts"
    )}
    assert {"analyze_activity_execution", "get_activity_best_efforts", "get_performance_curves"} <= deep

    assert [t.name for t in _select_planner_tools("create a workout for tomorrow")] == [
        "validate_and_preview_training_plan",
        "upload_training_plan_to_intervals",
    ]
    assert "update_calendar_event" in {
        t.name for t in _select_planner_tools("move my Thursday run to Friday")
    }
    library = {t.name for t in _select_planner_tools("save this workout to my library")}
    assert {"get_workout_library", "save_workouts_to_library"} <= library
    assert "delete_planned_event_from_intervals" in {
        t.name for t in _select_planner_tools("delete the workout on Friday")
    }


def test_performance_reading_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERVALS_ICU_API_KEY", "test_secret")
    import json as jsonlib

    from hybrid_training_agent.tools.analytics_tools import (
        analyze_activity_execution,
        get_activity_best_efforts,
        get_performance_curves,
        search_completed_activities,
        search_similar_interval_sessions,
    )

    with respx.mock(base_url="https://intervals.icu/api/v1") as respx_mock:
        pace_route = respx_mock.get("/athlete/0/pace-curves.json").mock(
            return_value=httpx.Response(200, json={"curves": [{"secs": 300, "value": 3.5}]})
        )
        hr_route = respx_mock.get("/athlete/0/hr-curves.json").mock(
            return_value=httpx.Response(200, json={"curves": [{"secs": 300, "value": 172}]})
        )
        search_route = respx_mock.get("/athlete/0/activities/search").mock(
            return_value=httpx.Response(200, json=[{"id": "i1", "name": "VO2max 5x1km"}])
        )
        interval_route = respx_mock.get("/athlete/0/activities/interval-search").mock(
            return_value=httpx.Response(200, json=[{"id": "i2", "name": "Threshold 4x8m"}])
        )
        intervals_route = respx_mock.get("/activity/i9/intervals").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "id": 1,
                        "type": "WORK",
                        "start_index": 0,
                        "end_index": 100,
                        "moving_time": 300,
                        "distance": 1000.0,
                        "average_heartrate": 170,
                        "max_heartrate": 178,
                        "average_speed": 3.5,
                        "intensity": 100,
                        "training_load": 20,
                        "zone": 4,
                    }
                ],
            )
        )
        streams_route = respx_mock.get("/activity/i9/streams.json").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {"type": "heartrate", "data": [150, 160, 170, 180]},
                    {"type": "velocity_smooth", "data": [3.2, 3.3, 3.4, 3.5]},
                ],
            )
        )
        best_route = respx_mock.get("/activity/i9/best-efforts").mock(
            return_value=httpx.Response(200, json=[{"start_index": 0, "value": 3.5}])
        )

        curves = get_performance_curves.invoke({"sport": "Run", "periods": "42d"})
        assert pace_route.called and hr_route.called
        assert curves["sport"] == "Run"

        found = search_completed_activities.invoke({"query": "VO2max"})
        assert search_route.called and found["count"] == 1

        similar = search_similar_interval_sessions.invoke(
            {
                "min_seconds": 280,
                "max_seconds": 320,
                "min_intensity_pct": 98,
                "max_intensity_pct": 102,
            }
        )
        assert interval_route.called and similar["count"] == 1

        execution = analyze_activity_execution.invoke({"activity_id": "i9"})
        assert intervals_route.called and streams_route.called
        assert execution["intervals"][0]["avg_hr"] == 170
        assert execution["intervals"][0]["avg_pace"] is not None
        assert execution["stream_summary"]["heartrate"]["avg"] == 165.0
        assert execution["stream_summary"]["pace_sec_per_km"]["avg"] is not None

        efforts = get_activity_best_efforts.invoke(
            {"activity_id": "i9", "duration_seconds": 300}
        )
        assert best_route.called and efforts == [{"start_index": 0, "value": 3.5}]


def test_workout_library_and_plan_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERVALS_ICU_API_KEY", "test_secret")
    from hybrid_training_agent.tools.planner_tools import (
        get_athlete_training_plan,
        get_workout_library,
    )

    with respx.mock(base_url="https://intervals.icu/api/v1") as respx_mock:
        respx_mock.get("/athlete/0/workouts").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "id": 1,
                        "name": "5x1km VO2max",
                        "type": "Run",
                        "folder_id": 5,
                        "moving_time": 2400,
                        "icu_training_load": 60,
                        "description": "- 1km 105% Pace",
                    }
                ],
            )
        )
        respx_mock.get("/athlete/0/folders").mock(
            return_value=httpx.Response(
                200, json=[{"id": 5, "name": "Base block", "type": "PLAN", "num_workouts": 12}]
            )
        )
        respx_mock.get("/athlete/0/training-plan").mock(
            return_value=httpx.Response(200, json={"id": "tp1", "training_plan_alias": "Half"})
        )

        library = get_workout_library.invoke({})
        assert library["workouts_count"] == 1
        assert library["folders_count"] == 1
        assert library["workouts"][0]["name"] == "5x1km VO2max"

        plan = get_athlete_training_plan.invoke({})
        assert plan["id"] == "tp1"


def test_writes_require_explicit_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERVALS_ICU_API_KEY", "test_secret")
    import json as jsonlib

    from hybrid_training_agent.tools.planner_tools import (
        apply_training_plan_folder,
        delete_planned_event_from_intervals,
        save_workouts_to_library,
        update_calendar_event,
        upload_training_plan_to_intervals,
    )

    run_workout = {
        "date": "2026-10-20",
        "name": "Easy Run",
        "sport_type": "Run",
        "description": "- 30m 70% Pace",
    }

    with respx.mock(base_url="https://intervals.icu/api/v1") as respx_mock:
        bulk = respx_mock.post("/athlete/0/events/bulk").mock(
            return_value=httpx.Response(200, json=[{"id": 1}])
        )
        get_event = respx_mock.get("/athlete/0/events/123").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": 123,
                    "name": "Old workout",
                    "start_date_local": "2026-10-20T00:00:00",
                    "category": "WORKOUT",
                    "type": "Run",
                    "description": "old",
                },
            )
        )
        delete_event = respx_mock.delete("/athlete/0/events/123").mock(
            return_value=httpx.Response(200, json={})
        )
        put_event = respx_mock.put("/athlete/0/events/123").mock(
            return_value=httpx.Response(200, json={"id": 123, "name": "New name"})
        )
        create_workout = respx_mock.post("/athlete/0/workouts").mock(
            return_value=httpx.Response(200, json={"id": 7})
        )
        apply_plan = respx_mock.post("/athlete/0/events/apply-plan").mock(
            return_value=httpx.Response(200, json={"applied": 12})
        )

        # 1. Upload
        preview = upload_training_plan_to_intervals.invoke({"workouts": [run_workout]})
        assert preview["status"] == "confirmation_required"
        assert not bulk.called
        uploaded = upload_training_plan_to_intervals.invoke(
            {"workouts": [run_workout], "confirm": True}
        )
        assert bulk.called and uploaded["status"] == "uploaded"

        # 2. Delete
        preview = delete_planned_event_from_intervals.invoke({"event_id": "123"})
        assert preview["status"] == "confirmation_required"
        assert preview["details"]["event"]["name"] == "Old workout"
        assert not delete_event.called
        deleted = delete_planned_event_from_intervals.invoke(
            {"event_id": "123", "confirm": True}
        )
        assert delete_event.called and deleted["status"] == "deleted"

        # 3. Update event
        preview = update_calendar_event.invoke({"event_id": "123", "updates": {"name": "New name"}})
        assert preview["status"] == "confirmation_required"
        assert preview["details"]["changes"]["name"] == {"from": "Old workout", "to": "New name"}
        assert not put_event.called
        updated = update_calendar_event.invoke(
            {"event_id": "123", "updates": {"name": "New name"}, "confirm": True}
        )
        assert put_event.called and updated["status"] == "updated"
        # Unsupported fields are rejected outright.
        rejected = update_calendar_event.invoke(
            {"event_id": "123", "updates": {"push_errors": []}, "confirm": True}
        )
        assert rejected["status"] == "failed"

        # 4. Save to library
        preview = save_workouts_to_library.invoke(
            {"workouts": [{"name": "Library Run", "sport_type": "Run", "description": "- 20m 70% Pace"}], "folder_id": 5}
        )
        assert preview["status"] == "confirmation_required"
        assert not create_workout.called
        saved = save_workouts_to_library.invoke(
            {
                "workouts": [
                    {"name": "Library Run", "sport_type": "Run", "description": "- 20m 70% Pace"}
                ],
                "folder_id": 5,
                "confirm": True,
            }
        )
        assert create_workout.called and saved["status"] == "saved"
        body = jsonlib.loads(create_workout.calls[0].request.content)
        assert body["folder_id"] == 5 and body["type"] == "Run"

        # 5. Apply plan folder
        preview = apply_training_plan_folder.invoke(
            {"folder_id": 9, "start_date_local": "2026-11-02"}
        )
        assert preview["status"] == "confirmation_required"
        assert not apply_plan.called
        applied = apply_training_plan_folder.invoke(
            {"folder_id": 9, "start_date_local": "2026-11-02", "confirm": True}
        )
        assert apply_plan.called and applied["status"] == "applied"


def test_planner_prompt_requires_confirmation_gate() -> None:
    from hybrid_training_agent.agents.prompts import PLANNER_SYSTEM_PROMPT

    lower = PLANNER_SYSTEM_PROMPT.lower()
    assert "confirmation gate" in lower
    assert "confirm=true" in lower
    assert "confirm=false" in lower
    for tool_name in (
        "upload_training_plan_to_intervals",
        "delete_planned_event_from_intervals",
        "update_calendar_event",
        "save_workouts_to_library",
        "apply_training_plan_folder",
    ):
        assert tool_name in PLANNER_SYSTEM_PROMPT


def test_analyst_prompt_lists_read_only_analysis_tools() -> None:
    from hybrid_training_agent.agents.prompts import ANALYST_SYSTEM_PROMPT

    for tool_name in (
        "get_performance_curves",
        "analyze_activity_execution",
        "search_similar_interval_sessions",
        "get_activity_best_efforts",
    ):
        assert tool_name in ANALYST_SYSTEM_PROMPT


def test_routing_eval_set() -> None:
    """Tier-1 heuristic regression set: intent -> expected pipeline (or ambiguity)."""
    import json
    from pathlib import Path

    cases = json.loads(
        (Path(__file__).parent / "routing_eval_cases.json").read_text(encoding="utf-8")
    )
    failures = []
    for case in cases:
        plan = classify_intent(case["prompt"], mode=case.get("mode", "auto"))
        if case.get("confident", True):
            if plan.pipeline != case["expected_pipeline"]:
                failures.append(
                    {
                        "prompt": case["prompt"],
                        "got": plan.pipeline,
                        "expected": case["expected_pipeline"],
                    }
                )
        elif plan.confident:
            failures.append({"prompt": case["prompt"], "got": "confident", "expected": "ambiguous"})
    assert not failures, failures


class _StubRouterLLM:
    """Minimal stand-in for a structured-output router model."""

    def __init__(self, decision: str, fail: bool = False) -> None:
        self.decision = decision
        self.fail = fail
        self.calls = 0

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        outer = self

        class _Structured:
            def invoke(self, messages: Any, config: Any = None, **kwargs: Any) -> Any:
                outer.calls += 1
                if outer.fail:
                    raise RuntimeError("router unavailable")
                from hybrid_training_agent.agents.graph import RouteDecision

                return RouteDecision(next_agent=outer.decision)

        return _Structured()


def test_confident_turn_never_calls_the_llm_router() -> None:
    llm = _RecordingFakeChatModel(messages=iter([AIMessage(content="Analyst: 25 km this week.")]))
    router = _StubRouterLLM("planner")
    graph = build_hybrid_training_graph(llm=llm, router_llm=router)
    state = graph.invoke(
        {
            "messages": [HumanMessage(content="how many kms did I run this week")],
            "visited_agents": [],
            "mode": "auto",
        }
    )
    assert state["visited_agents"] == ["analyst"]
    assert state["router"] == "heuristic"
    assert router.calls == 0


def test_llm_router_handles_ambiguous_follow_up() -> None:
    llm = _RecordingFakeChatModel(
        messages=iter([AIMessage(content="Planner: here is the 60m session.")])
    )
    router = _StubRouterLLM("planner")
    graph = build_hybrid_training_graph(llm=llm, router_llm=router)
    state = graph.invoke(
        {
            "messages": [HumanMessage(content="sounds good, go ahead")],
            "visited_agents": [],
            "mode": "auto",
        }
    )
    assert state["router"] == "llm"
    assert state["visited_agents"] == ["planner"]
    # Decided once, then asked again after the hop (and stopped since planner was already run).
    assert router.calls == 2


def test_supervisor_falls_back_to_heuristic_when_router_fails() -> None:
    llm = _RecordingFakeChatModel(
        messages=iter(
            [
                AIMessage(content="Analyst: overview."),
                AIMessage(content="Planner: plan."),
            ]
        )
    )
    router = _StubRouterLLM("planner", fail=True)
    graph = build_hybrid_training_graph(llm=llm, router_llm=router)
    state = graph.invoke(
        {
            "messages": [HumanMessage(content="ok do that")],
            "visited_agents": [],
            "mode": "auto",
        }
    )
    assert state["router"] == "heuristic-fallback"
    assert state["visited_agents"] == ["analyst", "planner"]
    assert router.calls == 1


def test_extract_message_text_from_content_blocks() -> None:
    blocks = [
        {"type": "thinking", "thinking": "Internal chain of thought..."},
        {"type": "text", "text": "### Hybrid Analysis\n- **CTL**: 52.0"},
        {"type": "tool_use", "id": "call_1", "name": "analyze_training_and_wellness", "input": {}},
    ]
    extracted = extract_message_text(blocks)
    assert extracted == "### Hybrid Analysis\n- **CTL**: 52.0"

    # Also test stringified JSON content blocks
    import json

    assert extract_message_text(json.dumps(blocks)) == "### Hybrid Analysis\n- **CTL**: 52.0"


def test_ttl_cache_expiry_and_disable() -> None:
    cache = TTLCache(ttl=10, max_entries=2)
    cache.set("k", {"v": 1})
    cached = cache.get("k")
    assert cached == {"v": 1}
    # Mutating a returned value must not poison the cache (deep copy on read).
    cached["v"] = 999
    assert cache.get("k") == {"v": 1}

    expired = TTLCache(ttl=0.05, max_entries=2)
    expired.set("k", 1)
    import time

    time.sleep(0.3)
    assert expired.get("k") is None

    disabled = TTLCache(ttl=0)
    disabled.set("k", 1)
    assert disabled.get("k") is None


def test_intervals_reads_are_cached_and_writes_invalidate() -> None:
    settings = Settings(intervals_api_key="test_secret", intervals_athlete_id="0")
    client = IntervalsIcuClient(settings=settings)
    with respx.mock(base_url="https://intervals.icu/api/v1") as respx_mock:
        profile_route = respx_mock.get("/athlete/0").mock(
            return_value=httpx.Response(200, json={"id": "0", "ctl": 60})
        )
        bulk_route = respx_mock.post("/athlete/0/events/bulk").mock(
            return_value=httpx.Response(200, json=[{"id": 1}])
        )

        assert client.get_athlete_profile()["ctl"] == 60
        assert client.get_athlete_profile()["ctl"] == 60
        assert profile_route.call_count == 1  # second read served from cache

        client.create_events_bulk(
            [
                PlannedWorkoutInput(
                    date="2026-10-20",
                    name="Session",
                    sport_type="Run",
                    description="- 30m 70% Pace",
                )
            ]
        )
        assert bulk_route.call_count == 1  # any write discards cached reads

        client.get_athlete_profile()
        assert profile_route.call_count == 2


def test_pubmed_search_results_are_cached() -> None:
    xml = (
        "<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>1</PMID>"
        "<Article><ArticleTitle>Cached study</ArticleTitle></Article>"
        "</MedlineCitation></PubmedArticle></PubmedArticleSet>"
    )
    client = ScienceLiteratureClient()
    with respx.mock(base_url="https://eutils.ncbi.nlm.nih.gov") as respx_mock:
        search_route = respx_mock.get("/entrez/eutils/esearch.fcgi").mock(
            return_value=httpx.Response(200, json={"esearchresult": {"idlist": ["1"]}})
        )
        fetch_route = respx_mock.get("/entrez/eutils/efetch.fcgi").mock(
            return_value=httpx.Response(200, text=xml)
        )

        first = client.search_pubmed("concurrent training", max_results=1)
        second = client.search_pubmed("concurrent training", max_results=1)

        assert first == second
        assert first[0]["title"] == "Cached study"
        assert search_route.call_count == 1
        assert fetch_route.call_count == 1


def test_detect_days_back_from_user_text() -> None:
    assert _detect_days_back("Create a 60m strength session", default=28) == 28
    assert _detect_days_back("Analyze my last 60 days please") == 60
    assert _detect_days_back("review the past 7 days") == 7
    # Out-of-range windows are clamped to a sane 7..365 range.
    assert _detect_days_back("last 2 days") == 7
    assert _detect_days_back("last 9999 days") == 365


def test_planner_tool_selection_binds_only_what_is_needed() -> None:
    names = [t.name for t in _select_planner_tools("Create a 60m full body strength session")]
    assert names == ["validate_and_preview_training_plan", "upload_training_plan_to_intervals"]
    # The pre-fetched calendar and curated evidence tools are never bound.
    assert "get_upcoming_intervals_calendar" not in names
    assert "get_curated_hybrid_training_evidence" not in names

    delete_names = [t.name for t in _select_planner_tools("Delete the workout on Friday")]
    assert "delete_planned_event_from_intervals" in delete_names


def test_analyst_node_prefetches_data_and_skips_tool_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERVALS_ICU_API_KEY", "test_key")
    calls: dict[str, Any] = {}

    class _StubTool:
        def __init__(self, name: str, result: Any) -> None:
            self.name = name
            self.result = result

        def invoke(self, args: dict[str, Any]) -> Any:
            calls[self.name] = args
            return self.result

    monkeypatch.setattr(
        graph_module,
        "fetch_athlete_profile_and_zones",
        _StubTool("fetch_athlete_profile_and_zones", {"name": "Alex", "ctl": 60}),
    )
    monkeypatch.setattr(
        graph_module,
        "analyze_training_and_wellness",
        _StubTool("analyze_training_and_wellness", {"load_and_form": {"ctl_fitness": 60.0}}),
    )

    seen_systems: list[str] = []

    class RecordingChatModel(GenericFakeChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self

        def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
            seen_systems.append(str(input[0].content))
            return super().invoke(input, config=config, **kwargs)

    fake_llm = RecordingChatModel(messages=iter([AIMessage(content="Analyst Report: ready.")]))
    graph = build_hybrid_training_graph(llm=fake_llm)
    graph.invoke(
        {
            "messages": [HumanMessage(content="Analyze my last 60 days on intervals.icu")],
            "next_agent": "supervisor",
            "visited_agents": [],
            "mode": "analyze",
        }
    )

    # Data was fetched deterministically (once) and the window came from the user's text.
    assert calls["analyze_training_and_wellness"] == {"days_back": 60}
    assert "fetch_athlete_profile_and_zones" in calls
    assert seen_systems and "Pre-fetched tool results" in seen_systems[0]
    assert "Alex" in seen_systems[0]


def test_analyst_node_reports_missing_credentials_without_prefetch() -> None:
    seen_systems: list[str] = []

    class RecordingChatModel(GenericFakeChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self

        def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
            seen_systems.append(str(input[0].content))
            return super().invoke(input, config=config, **kwargs)

    fake_llm = RecordingChatModel(messages=iter([AIMessage(content="No data available.")]))
    graph = build_hybrid_training_graph(llm=fake_llm)
    graph.invoke(
        {
            "messages": [HumanMessage(content="Analyze my intervals.icu data")],
            "next_agent": "supervisor",
            "visited_agents": [],
            "mode": "analyze",
        }
    )

    assert seen_systems and "credentials are not configured" in seen_systems[0]



