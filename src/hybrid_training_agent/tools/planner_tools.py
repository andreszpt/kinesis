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


def _confirmation_required(
    action: str,
    summary: str,
    details: dict[str, Any],
    instruction: str | None = None,
) -> dict[str, Any]:
    """Standard payload returned by mutating tools until the user confirms."""
    return {
        "status": "confirmation_required",
        "action": action,
        "summary": summary,
        "details": details,
        "instruction": instruction
        or (
            "Show this preview to the user and ask for explicit confirmation. "
            "Only call this tool again with confirm=true after the user clearly agrees."
        ),
    }


# Event fields the agent is allowed to change on the intervals.icu calendar.
UPDATABLE_EVENT_FIELDS = {
    "name",
    "description",
    "start_date_local",
    "end_date_local",
    "category",
    "type",
    "moving_time",
    "icu_training_load",
    "color",
    "show_as_note",
    "not_on_fitness_chart",
    "distance",
    "target",
    "tags",
    "indoor",
    "carbs_per_hour",
}


@tool
def validate_and_preview_training_plan(
    workouts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate a proposed hybrid training plan before uploading to intervals.icu.

    Checks:
    1. Every 'Run' workout uses valid intervals.icu step syntax (e.g. `- 10m 65-75% Pace`) so it syncs as a structured workout to Suunto.
    2. Every 'WeightTraining' workout is formatted as a readable text workout and uploaded as an intervals.icu calendar note (no Suunto step syntax, no watch guide push).
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
    confirm: bool = False,
) -> dict[str, Any]:
    """Upload a list of planned Running and Strength workouts (or Notes) to intervals.icu calendar.

    Uses `POST /api/v1/athlete/{id}/events/bulk` with `upsert=true` so re-uploading updates existing events cleanly.
    Running workouts formatted with intervals.icu step syntax (`- 10m 70-78% Pace`) will automatically sync to Suunto,
    while WeightTraining workouts are uploaded as readable calendar notes so intervals.icu never attempts a Suunto
    guide push for them (which would fail for a step-less strength workout).

    SAFETY: call first with `confirm=false` (default) to get a preview. You MUST show that preview to the user and
    ask for explicit confirmation. Only call again with `confirm=true` after the user clearly agrees.
    """
    parsed = [PlannedWorkoutInput(**w) for w in workouts]
    audit = audit_planned_workouts(parsed)
    payloads = [IntervalsIcuClient.build_event_payload(w) for w in parsed]

    if not confirm:
        return _confirmation_required(
            "upload_training_plan_to_intervals",
            f"Upload/update {len(parsed)} workout(s) on the intervals.icu calendar (upsert={upsert}).",
            {
                "upsert": upsert,
                "workouts": [w.model_dump() for w in parsed],
                "intervals_event_payloads_preview": payloads,
                "audit": audit,
            },
        )

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
def delete_planned_event_from_intervals(
    event_id: str, confirm: bool = False
) -> dict[str, Any]:
    """Delete a planned workout or note from the athlete's intervals.icu calendar by event ID.

    SAFETY: call first with `confirm=false` (default) to get a preview of the event. You MUST show it to the user
    and ask for explicit confirmation. Only call again with `confirm=true` after the user clearly agrees.
    """
    client = IntervalsIcuClient()
    if not confirm:
        try:
            event = client.get_event(event_id)
        except IntervalsIcuError as exc:
            return {"status": "failed", "event_id": event_id, "error": str(exc)}
        preview = {
            key: event.get(key)
            for key in ("id", "name", "start_date_local", "category", "type", "description")
        }
        return _confirmation_required(
            "delete_planned_event_from_intervals",
            f"Delete calendar event {event_id} ('{event.get('name')}' on {event.get('start_date_local')}).",
            {"event": preview},
        )

    try:
        client.delete_event(event_id)
        return {"status": "deleted", "event_id": event_id}
    except IntervalsIcuError as exc:
        return {"status": "failed", "event_id": event_id, "error": str(exc)}


@tool
def update_calendar_event(
    event_id: str, updates: dict[str, Any], confirm: bool = False
) -> dict[str, Any]:
    """Update a single planned workout/note on the intervals.icu calendar (move, rename, edit, change load).

    SAFETY: call first with `confirm=false` (default) to get a before/after preview. You MUST show it to the user
    and ask for explicit confirmation. Only call again with `confirm=true` after the user clearly agrees.
    """
    invalid = sorted(set(updates) - UPDATABLE_EVENT_FIELDS)
    if invalid:
        return {
            "status": "failed",
            "error": f"Unsupported event fields: {', '.join(invalid)}",
            "allowed_fields": sorted(UPDATABLE_EVENT_FIELDS),
        }

    client = IntervalsIcuClient()
    try:
        current = client.get_event(event_id)
    except IntervalsIcuError as exc:
        return {"status": "failed", "event_id": event_id, "error": str(exc)}

    if not confirm:
        changes = {
            key: {"from": current.get(key), "to": value}
            for key, value in updates.items()
        }
        return _confirmation_required(
            "update_calendar_event",
            f"Update calendar event {event_id} ('{current.get('name')}'): {', '.join(sorted(updates))}.",
            {"event_id": event_id, "changes": changes},
        )

    try:
        updated = client.update_event(event_id, updates)
        return {"status": "updated", "event_id": event_id, "event": updated}
    except IntervalsIcuError as exc:
        return {"status": "failed", "event_id": event_id, "error": str(exc)}


# ---------------------------------------------------------------------------
# Workout library & plans
# ---------------------------------------------------------------------------
@tool
def get_workout_library() -> dict[str, Any]:
    """List the athlete's saved workouts, folders and plans from the intervals.icu library.

    Use this before saving a new workout (to avoid duplicates) or to find a plan folder to apply.
    """
    client = IntervalsIcuClient()
    try:
        workouts = client.get_workouts()
        folders = client.get_folders()
    except IntervalsIcuError as exc:
        return {"error": str(exc)}

    workout_summaries = [
        {
            key: w.get(key)
            for key in ("id", "name", "type", "folder_id", "moving_time", "icu_training_load", "description")
        }
        for w in workouts
    ]
    folder_summaries = [
        {
            key: f.get(key)
            for key in ("id", "name", "type", "description", "num_workouts", "duration_weeks")
        }
        for f in folders
    ]
    return {
        "workouts_count": len(workout_summaries),
        "folders_count": len(folder_summaries),
        "workouts": workout_summaries,
        "folders": folder_summaries,
    }


@tool
def get_athlete_training_plan() -> dict[str, Any]:
    """Get the athlete's currently applied intervals.icu training plan."""
    client = IntervalsIcuClient()
    try:
        return client.get_training_plan()
    except IntervalsIcuError as exc:
        return {"error": str(exc)}


@tool
def save_workouts_to_library(
    workouts: list[dict[str, Any]],
    folder_id: int | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """Save workouts into the athlete's intervals.icu library, optionally inside a folder/plan.

    Each workout accepts: name, sport_type (Run/WeightTraining), description, moving_time_minutes,
    estimated_load, day (0-based day inside a plan folder), folder_id (per-workout override).

    SAFETY: call first with `confirm=false` (default) to get a preview. You MUST show it to the user and ask for
    explicit confirmation. Only call again with `confirm=true` after the user clearly agrees.
    """
    payloads: list[dict[str, Any]] = []
    for workout in workouts:
        sport = str(workout.get("sport_type") or workout.get("type") or "Run")
        target_folder = workout.get("folder_id", folder_id)
        payload: dict[str, Any] = {
            "name": workout.get("name") or "Untitled workout",
            "type": sport,
            "description": workout.get("description") or "",
        }
        if target_folder is not None:
            payload["folder_id"] = int(target_folder)
        if workout.get("moving_time_minutes") is not None:
            payload["moving_time"] = int(float(workout["moving_time_minutes"]) * 60)
        if workout.get("estimated_load") is not None:
            payload["icu_training_load"] = int(workout["estimated_load"])
        if workout.get("day") is not None:
            payload["day"] = int(workout["day"])
        payloads.append(payload)

    if not confirm:
        return _confirmation_required(
            "save_workouts_to_library",
            f"Save {len(payloads)} workout(s) to the intervals.icu library"
            + (f" (folder {folder_id})." if folder_id is not None else "."),
            {"workouts": payloads},
        )

    client = IntervalsIcuClient()
    created: list[dict[str, Any]] = []
    try:
        for payload in payloads:
            created.append(client.create_workout(payload))
        return {"status": "saved", "saved_count": len(created), "workouts": created}
    except IntervalsIcuError as exc:
        return {"status": "failed", "error": str(exc), "saved": created}


@tool
def apply_training_plan_folder(
    folder_id: int, start_date_local: str, confirm: bool = False
) -> dict[str, Any]:
    """Apply a workout folder/plan from the library to the athlete's calendar starting on a date.

    SAFETY: call first with `confirm=false` (default) to get a preview. You MUST show it to the user and ask for
    explicit confirmation. Only call again with `confirm=true` after the user clearly agrees.
    """
    if not confirm:
        return _confirmation_required(
            "apply_training_plan_folder",
            f"Apply plan folder {folder_id} to the calendar starting {start_date_local}.",
            {"folder_id": int(folder_id), "start_date_local": start_date_local},
        )

    client = IntervalsIcuClient()
    try:
        result = client.apply_plan(int(folder_id), start_date_local)
        return {
            "status": "applied",
            "folder_id": int(folder_id),
            "start_date_local": start_date_local,
            "result": result,
        }
    except IntervalsIcuError as exc:
        return {"status": "failed", "error": str(exc)}


PLANNER_CORE_TOOLS = [
    validate_and_preview_training_plan,
    upload_training_plan_to_intervals,
]

PLANNER_LIBRARY_TOOLS = [
    get_workout_library,
    get_athlete_training_plan,
    save_workouts_to_library,
    apply_training_plan_folder,
]

PLANNER_TOOLS = [
    get_upcoming_intervals_calendar,
    validate_and_preview_training_plan,
    upload_training_plan_to_intervals,
    delete_planned_event_from_intervals,
    update_calendar_event,
    *PLANNER_LIBRARY_TOOLS,
]
