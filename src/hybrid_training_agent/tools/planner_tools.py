from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from langchain_core.tools import tool

from hybrid_training_agent.clients.intervals_client import (
    IntervalsIcuClient,
    IntervalsIcuError,
    validate_intervals_run_syntax,
)
from hybrid_training_agent.models import PlannedWorkoutInput


def audit_planned_workouts(workouts: list[PlannedWorkoutInput]) -> dict[str, Any]:
    """Validate intervals.icu workout syntax for Suunto running sync and audit concurrent interference spacing."""
    syntax_checks: list[dict[str, Any]] = []
    interference_warnings: list[str] = []
    total_run_minutes = 0
    total_strength_minutes = 0
    total_estimated_load = 0

    by_date: dict[str, list[PlannedWorkoutInput]] = {}
    for w in workouts:
        day_str = w.date[:10]
        by_date.setdefault(day_str, []).append(w)

        if w.moving_time_minutes:
            if w.sport_type == "Run":
                total_run_minutes += w.moving_time_minutes
            elif w.sport_type == "WeightTraining":
                total_strength_minutes += w.moving_time_minutes
        if w.estimated_load:
            total_estimated_load += w.estimated_load

        if w.sport_type == "Run":
            check = validate_intervals_run_syntax(w.description)
            syntax_checks.append(
                {
                    "date": day_str,
                    "name": w.name,
                    "sport_type": w.sport_type,
                    **check,
                }
            )
        elif w.sport_type == "WeightTraining":
            syntax_checks.append(
                {
                    "date": day_str,
                    "name": w.name,
                    "sport_type": w.sport_type,
                    "is_valid_structured_workout": bool(w.description.strip()),
                    "issues": [] if w.description.strip() else ["WeightTraining description is empty."],
                }
            )

    sorted_days = sorted(by_date.keys())
    for idx, day_str in enumerate(sorted_days):
        day_workouts = by_date[day_str]
        types = [w.sport_type for w in day_workouts]
        if "Run" in types and "WeightTraining" in types:
            interference_warnings.append(
                f"Same-day Run + Strength on {day_str}: Ensure >=6 hours separation between sessions (Robineau et al., 2016) and perform the priority session first."
            )

        # Check if heavy lower-body strength is scheduled the day before a high-intensity interval or long run
        has_lower_strength = any(
            w.sport_type == "WeightTraining"
            and any(
                kw in (w.name + " " + w.description).lower()
                for kw in ("lower", "squat", "deadlift", "leg", "heavy")
            )
            for w in day_workouts
        )
        if has_lower_strength and idx + 1 < len(sorted_days):
            next_day = sorted_days[idx + 1]
            try:
                d1 = date.fromisoformat(day_str)
                d2 = date.fromisoformat(next_day)
                if (d2 - d1).days == 1:
                    for nw in by_date[next_day]:
                        if nw.sport_type == "Run" and any(
                            kw in (nw.name + " " + nw.description).lower()
                            for kw in ("interval", "vo2", "threshold", "tempo", "long", "track", "speed")
                        ):
                            interference_warnings.append(
                                f"Sequential Interference Caution ({day_str} -> {next_day}): Lower-body strength '{day_workouts[0].name}' "
                                f"is scheduled the day before quality/long run '{nw.name}'. Residual eccentric DOMS/CNS fatigue may impair running economy (Wilson et al., 2012; Blagrove et al., 2018)."
                            )
            except ValueError:
                pass

    all_valid = all(c.get("is_valid_structured_workout", True) for c in syntax_checks)
    return {
        "all_workouts_valid": all_valid,
        "total_workouts": len(workouts),
        "total_run_minutes": total_run_minutes,
        "total_strength_minutes": total_strength_minutes,
        "total_estimated_load": total_estimated_load,
        "syntax_validation": syntax_checks,
        "interference_audit_warnings": interference_warnings,
    }


@tool
def get_upcoming_intervals_calendar(days_ahead: int = 14) -> dict[str, Any]:
    """Fetch upcoming planned workouts and notes from the athlete's intervals.icu calendar."""
    client = IntervalsIcuClient()
    today = date.today()
    newest = (today + timedelta(days=days_ahead)).isoformat()
    try:
        events = client.get_events(oldest=today.isoformat(), newest=newest)
        return {
            "oldest": today.isoformat(),
            "newest": newest,
            "count": len(events),
            "events": events,
        }
    except IntervalsIcuError as exc:
        return {"error": str(exc), "events": []}


@tool
def validate_and_preview_training_plan(
    workouts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate a proposed hybrid training plan before uploading to intervals.icu.

    Checks:
    1. Every 'Run' workout uses valid intervals.icu step syntax (e.g. `- 10m 65-75% Pace`) so it syncs as a structured workout to Suunto.
    2. Every 'WeightTraining' workout is formatted as a readable text workout for intervals.icu (does not need Suunto step syntax).
    3. Audits concurrent training spacing (6-24h rule between heavy lower-body lifting and quality/long runs).

    Each item in `workouts` must have:
    - `date` (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
    - `name` (str)
    - `sport_type` ('Run', 'WeightTraining', or 'Note')
    - `description` (str)
    - `moving_time_minutes` (int, optional)
    - `estimated_load` (int, optional)
    """
    parsed = [PlannedWorkoutInput(**w) for w in workouts]
    audit = audit_planned_workouts(parsed)
    preview_payloads = [IntervalsIcuClient.build_event_payload(w) for w in parsed]
    return {
        "audit": audit,
        "intervals_event_payloads_preview": preview_payloads,
    }


@tool
def upload_training_plan_to_intervals(
    workouts: list[dict[str, Any]],
    upsert: bool = True,
) -> dict[str, Any]:
    """Upload a list of planned Running and Strength workouts (or Notes) to intervals.icu calendar.

    Uses `POST /api/v1/athlete/{id}/events/bulk` with `upsert=true` so re-uploading updates existing events cleanly.
    Running workouts formatted with intervals.icu step syntax (`- 10m 70-78% Pace`) will automatically sync to Suunto,
    while WeightTraining workouts are written directly onto the intervals.icu calendar.
    """
    parsed = [PlannedWorkoutInput(**w) for w in workouts]
    audit = audit_planned_workouts(parsed)
    client = IntervalsIcuClient()
    try:
        created = client.create_events_bulk(parsed, upsert=upsert)
        return {
            "status": "uploaded",
            "uploaded_count": len(created),
            "audit_summary": audit,
            "events": created,
        }
    except IntervalsIcuError as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "audit_summary": audit,
        }


@tool
def delete_planned_event_from_intervals(event_id: str) -> dict[str, Any]:
    """Delete a planned workout or note from the athlete's intervals.icu calendar by event ID."""
    client = IntervalsIcuClient()
    try:
        client.delete_event(event_id)
        return {"status": "deleted", "event_id": event_id}
    except IntervalsIcuError as exc:
        return {"status": "failed", "event_id": event_id, "error": str(exc)}


PLANNER_TOOLS = [
    get_upcoming_intervals_calendar,
    validate_and_preview_training_plan,
    upload_training_plan_to_intervals,
    delete_planned_event_from_intervals,
]
