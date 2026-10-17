from __future__ import annotations

from typing import Any

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _MCPServer
except ModuleNotFoundError:  # mcp 2.x renamed FastMCP -> MCPServer
    from mcp.server.mcpserver import MCPServer as _MCPServer

from hybrid_training_agent.tools.analytics_tools import (
    analyze_activity_execution,
    analyze_training_and_wellness,
    fetch_athlete_profile_and_zones,
    get_activity_best_efforts,
    get_activity_interval_stats,
    get_performance_curves,
    search_completed_activities,
    search_similar_interval_sessions,
)
from hybrid_training_agent.tools.planner_tools import (
    apply_training_plan_folder,
    delete_planned_event_from_intervals,
    get_athlete_training_plan,
    get_upcoming_intervals_calendar,
    get_workout_library,
    save_workouts_to_library,
    update_calendar_event,
    upload_training_plan_to_intervals,
    validate_and_preview_training_plan,
)
from hybrid_training_agent.tools.science_tools import (
    fetch_pubmed_papers_by_pmid,
    get_curated_hybrid_training_evidence,
    search_scientific_literature,
)

mcp = _MCPServer("kinesis")


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
def mcp_get_performance_curves(sport: str = "Run", periods: str = "42d,1y") -> dict[str, Any]:
    """Fetch the athlete's best pace and heart-rate curves for benchmarking (42d/1y/all/season)."""
    return get_performance_curves.invoke({"sport": sport, "periods": periods})


@mcp.tool()
def mcp_search_completed_activities(query: str, limit: int = 20) -> dict[str, Any]:
    """Search completed activities by name, or exact tag when the query starts with '#'."""
    return search_completed_activities.invoke({"query": query, "limit": limit})


@mcp.tool()
def mcp_search_similar_interval_sessions(
    min_seconds: int,
    max_seconds: int,
    min_intensity_pct: float,
    max_intensity_pct: float,
    sport: str = "Run",
    min_reps: int | None = None,
    max_reps: int | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Find completed activities with intervals matching a duration (s) and intensity (%) band."""
    return search_similar_interval_sessions.invoke(
        {
            "min_seconds": min_seconds,
            "max_seconds": max_seconds,
            "min_intensity_pct": min_intensity_pct,
            "max_intensity_pct": max_intensity_pct,
            "sport": sport,
            "min_reps": min_reps,
            "max_reps": max_reps,
            "limit": limit,
        }
    )


@mcp.tool()
def mcp_analyze_activity_execution(
    activity_id: str,
    include_streams: bool = True,
) -> dict[str, Any]:
    """Review how a completed session was executed: intervals/reps plus a compact stream summary."""
    return analyze_activity_execution.invoke(
        {"activity_id": activity_id, "include_streams": include_streams}
    )


@mcp.tool()
def mcp_get_activity_interval_stats(
    activity_id: str, start_index: int, end_index: int
) -> dict[str, Any]:
    """Compute interval-like stats for an arbitrary slice of an activity's streams."""
    return get_activity_interval_stats.invoke(
        {"activity_id": activity_id, "start_index": start_index, "end_index": end_index}
    )


@mcp.tool()
def mcp_get_activity_best_efforts(
    activity_id: str,
    duration_seconds: int | None = None,
    distance_meters: float | None = None,
    stream: str = "velocity_smooth",
    count: int = 5,
) -> dict[str, Any]:
    """Find the best efforts inside a completed activity for a duration or distance."""
    return get_activity_best_efforts.invoke(
        {
            "activity_id": activity_id,
            "duration_seconds": duration_seconds,
            "distance_meters": distance_meters,
            "stream": stream,
            "count": count,
        }
    )


@mcp.tool()
def mcp_get_workout_library() -> dict[str, Any]:
    """List saved workouts, folders and plans in the athlete's intervals.icu library."""
    return get_workout_library.invoke({})


@mcp.tool()
def mcp_get_athlete_training_plan() -> dict[str, Any]:
    """Get the athlete's currently applied intervals.icu training plan."""
    return get_athlete_training_plan.invoke({})


@mcp.tool()
def mcp_upload_training_plan_to_intervals(
    workouts: list[dict[str, Any]],
    upsert: bool = True,
    confirm: bool = False,
) -> dict[str, Any]:
    """Upload planned Running and Strength workouts to intervals.icu calendar.

    Call with confirm=false first to get a preview; only pass confirm=true after the user agrees.
    """
    return upload_training_plan_to_intervals.invoke(
        {"workouts": workouts, "upsert": upsert, "confirm": confirm}
    )


@mcp.tool()
def mcp_delete_planned_event_from_intervals(
    event_id: str, confirm: bool = False
) -> dict[str, Any]:
    """Delete a planned workout or note from intervals.icu by event ID.

    Call with confirm=false first to get a preview; only pass confirm=true after the user agrees.
    """
    return delete_planned_event_from_intervals.invoke(
        {"event_id": event_id, "confirm": confirm}
    )


@mcp.tool()
def mcp_update_calendar_event(
    event_id: str, updates: dict[str, Any], confirm: bool = False
) -> dict[str, Any]:
    """Update one planned calendar event (move, reschedule, rename, edit description/load).

    Call with confirm=false first to get a before/after preview; only pass confirm=true after the user agrees.
    """
    return update_calendar_event.invoke(
        {"event_id": event_id, "updates": updates, "confirm": confirm}
    )


@mcp.tool()
def mcp_save_workouts_to_library(
    workouts: list[dict[str, Any]],
    folder_id: int | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """Save workouts to the athlete's intervals.icu library (optionally inside a folder/plan).

    Call with confirm=false first to get a preview; only pass confirm=true after the user agrees.
    """
    return save_workouts_to_library.invoke(
        {"workouts": workouts, "folder_id": folder_id, "confirm": confirm}
    )


@mcp.tool()
def mcp_apply_training_plan_folder(
    folder_id: int, start_date_local: str, confirm: bool = False
) -> dict[str, Any]:
    """Apply a saved plan folder to the athlete's calendar starting on a date.

    Call with confirm=false first to get a preview; only pass confirm=true after the user agrees.
    """
    return apply_training_plan_folder.invoke(
        {"folder_id": folder_id, "start_date_local": start_date_local, "confirm": confirm}
    )


def run_mcp_server() -> None:
    mcp.run()


if __name__ == "__main__":
    run_mcp_server()
