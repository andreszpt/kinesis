from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from hybrid_training_agent.tools.analytics_tools import (
    analyze_training_and_wellness,
    fetch_athlete_profile_and_zones,
)
from hybrid_training_agent.tools.planner_tools import (
    delete_planned_event_from_intervals,
    get_upcoming_intervals_calendar,
    upload_training_plan_to_intervals,
    validate_and_preview_training_plan,
)
from hybrid_training_agent.tools.science_tools import (
    fetch_pubmed_papers_by_pmid,
    get_curated_hybrid_training_evidence,
    search_scientific_literature,
)

mcp = FastMCP("kinesis")


@mcp.tool()
def mcp_search_scientific_literature(
    query: str,
    max_results: int = 5,
    prefer_meta_analyses_and_trials: bool = True,
) -> dict[str, Any]:
    """Search PubMed and Europe PMC for peer-reviewed exercise physiology and sports science literature."""
    return search_scientific_literature.invoke(
        {
            "query": query,
            "max_results": max_results,
            "prefer_meta_analyses_and_trials": prefer_meta_analyses_and_trials,
        }
    )


@mcp.tool()
def mcp_fetch_pubmed_by_pmid(pmids: list[str]) -> dict[str, Any]:
    """Fetch complete metadata, abstract, and DOI for specific PubMed IDs (PMIDs)."""
    return fetch_pubmed_papers_by_pmid.invoke({"pmids": pmids})


@mcp.tool()
def mcp_get_curated_hybrid_evidence(topic: str = "all") -> dict[str, Any]:
    """Retrieve curated landmark meta-analyses and RCTs on hybrid running + strength training."""
    return get_curated_hybrid_training_evidence.invoke({"topic": topic})


@mcp.tool()
def mcp_fetch_athlete_profile() -> dict[str, Any]:
    """Fetch intervals.icu athlete profile, running threshold pace, LTHR, max HR, and training zones."""
    return fetch_athlete_profile_and_zones.invoke({})


@mcp.tool()
def mcp_analyze_training_and_wellness(days_back: int = 28) -> dict[str, Any]:
    """Analyze past training and health data (CTL, ATL, TSB, ACWR, HRV, RHR, Sleep, Running polarization, and Strength sessions) from intervals.icu."""
    return analyze_training_and_wellness.invoke({"days_back": days_back})


@mcp.tool()
def mcp_get_upcoming_intervals_calendar(days_ahead: int = 14) -> dict[str, Any]:
    """Fetch upcoming planned workouts and notes from the athlete's intervals.icu calendar."""
    return get_upcoming_intervals_calendar.invoke({"days_ahead": days_ahead})


@mcp.tool()
def mcp_validate_and_preview_training_plan(
    workouts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate intervals.icu workout syntax for Suunto running sync and audit concurrent interference spacing."""
    return validate_and_preview_training_plan.invoke({"workouts": workouts})


@mcp.tool()
def mcp_upload_training_plan_to_intervals(
    workouts: list[dict[str, Any]],
    upsert: bool = True,
) -> dict[str, Any]:
    """Upload planned Running and Strength workouts to intervals.icu calendar."""
    return upload_training_plan_to_intervals.invoke(
        {"workouts": workouts, "upsert": upsert}
    )


@mcp.tool()
def mcp_delete_planned_event_from_intervals(event_id: str) -> dict[str, Any]:
    """Delete a planned workout or note from intervals.icu by event ID."""
    return delete_planned_event_from_intervals.invoke({"event_id": event_id})


def run_mcp_server() -> None:
    mcp.run()


if __name__ == "__main__":
    run_mcp_server()
