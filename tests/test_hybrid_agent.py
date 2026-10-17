from __future__ import annotations

import io
from typing import Any

import httpx
import pytest
import respx
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from hybrid_training_agent.agents.graph import (
    build_hybrid_training_graph,
    classify_intent_pipeline,
)
from hybrid_training_agent import cli
from hybrid_training_agent.cli import _pending_question, extract_message_text
from hybrid_training_agent.clients.intervals_client import (
    IntervalsIcuClient,
    validate_intervals_run_syntax,
)
from hybrid_training_agent.clients.science_client import ScienceLiteratureClient
from hybrid_training_agent.config import Settings, get_chat_model
from hybrid_training_agent.models import PlannedWorkoutInput
from hybrid_training_agent.tools.analytics_tools import compute_hybrid_analytics
from hybrid_training_agent.tools.planner_tools import audit_planned_workouts


class _RecordingFakeChatModel(GenericFakeChatModel):
    """Scripted fake chat model that plays nicely with `bind_tools`."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


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


