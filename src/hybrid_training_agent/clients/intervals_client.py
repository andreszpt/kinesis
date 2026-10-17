from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any

import httpx

from hybrid_training_agent.config import Settings, get_settings
from hybrid_training_agent.models import PlannedWorkoutInput


class IntervalsIcuError(Exception):
    """Raised when an intervals.icu API request fails or credentials are missing."""


# Regex matching valid intervals.icu workout builder step lines, e.g.:
# - 10m 65-75% Pace
# - 400mtr 105% Pace
# - 5m Z2 HR
# - 30s 95% LTHR
_STEP_LINE_RE = re.compile(
    r"^\s*-\s+.*?(?:\d+\s*(?:m|min|mins|s|sec|secs|h|hr|km|mtr|m\b)|\d+:\d+)",
    re.IGNORECASE,
)
_TARGET_RE = re.compile(
    r"(?:%\s*(?:pace|lthr|hr|mmp|ftp)|z[1-7]|\bhr\b|\bpace\b|\brpe\b)",
    re.IGNORECASE,
)


def validate_intervals_run_syntax(description: str) -> dict[str, Any]:
    """Validate that a Run workout description uses intervals.icu step syntax for Suunto sync.

    Intervals.icu parses lines starting with `- ` (e.g., `- 15m 70-78% Pace` or `- 1km 98-102% Pace`)
    into structured watch steps that push directly to Suunto, Garmin, Coros, and Wahoo.
    """
    lines = [line.strip() for line in description.splitlines() if line.strip()]
    step_lines = [line for line in lines if line.startswith("-")]
    valid_steps = [line for line in step_lines if _STEP_LINE_RE.match(line)]
    steps_with_targets = [line for line in valid_steps if _TARGET_RE.search(line)]

    issues: list[str] = []
    if not step_lines:
        issues.append(
            "No workout step lines starting with '- ' found. For Suunto watch sync, include lines like '- 10m 65-75% Pace' or '- 5m Z2 Pace'."
        )
    elif not valid_steps:
        issues.append(
            "Step lines starting with '- ' were found, but none contained a recognizable duration/distance (e.g., '10m', '30s', '1km', '400mtr')."
        )
    elif not steps_with_targets:
        issues.append(
            "Workout steps have durations/distances, but lack intensity targets (e.g., '70-80% Pace', 'Z2 Pace', '85% LTHR'). Adding targets ensures structured guidance on Suunto."
        )

    return {
        "is_valid_structured_workout": len(valid_steps) > 0 and len(issues) == 0,
        "total_step_lines": len(step_lines),
        "valid_duration_steps": len(valid_steps),
        "steps_with_intensity_targets": len(steps_with_targets),
        "issues": issues,
    }


class IntervalsIcuClient:
    """HTTP client for the intervals.icu REST API v1."""

    def __init__(self, settings: Settings | None = None, timeout: float = 30.0) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.intervals_base_url
        self.athlete_id = self.settings.intervals_athlete_id or "0"
        self.timeout = timeout

    def _require_credentials(self) -> tuple[str, str]:
        if not self.settings.intervals_api_key:
            raise IntervalsIcuError(
                "INTERVALS_ICU_API_KEY is not configured. Set INTERVALS_ICU_API_KEY in your .env file "
                "(generate your personal API key in intervals.icu -> Settings -> Developer Settings)."
            )
        return ("API_KEY", self.settings.intervals_api_key)

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
    ) -> Any:
        auth = self._require_credentials()
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            with httpx.Client(auth=auth, timeout=self.timeout) as client:
                response = client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_body,
                    headers={"Accept": "application/json"},
                )
                response.raise_for_status()
                if not response.content:
                    return {}
                return response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500] if exc.response else str(exc)
            raise IntervalsIcuError(
                f"Intervals.icu API error ({exc.response.status_code}) on {method} {path}: {detail}"
            ) from exc
        except httpx.RequestError as exc:
            raise IntervalsIcuError(
                f"Network error connecting to intervals.icu ({method} {path}): {exc}"
            ) from exc

    def get_athlete_profile(self) -> dict[str, Any]:
        """Fetch athlete profile, threshold settings, HR zones, and pace zones."""
        return self._request("GET", f"athlete/{self.athlete_id}")

    def get_wellness(
        self,
        oldest: str | date | None = None,
        newest: str | date | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch daily wellness records (CTL, ATL, rampRate, HRV, RHR, sleep, soreness, fatigue)."""
        today = date.today()
        oldest_str = (
            oldest.isoformat()
            if isinstance(oldest, date)
            else (oldest or (today - timedelta(days=28)).isoformat())
        )
        newest_str = (
            newest.isoformat()
            if isinstance(newest, date)
            else (newest or today.isoformat())
        )
        data = self._request(
            "GET",
            f"athlete/{self.athlete_id}/wellness",
            params={"oldest": oldest_str, "newest": newest_str},
        )
        return data if isinstance(data, list) else []

    def get_activities(
        self,
        oldest: str | date | None = None,
        newest: str | date | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch completed activities in [oldest, newest] date range."""
        today = date.today()
        oldest_str = (
            oldest.isoformat()
            if isinstance(oldest, date)
            else (oldest or (today - timedelta(days=28)).isoformat())
        )
        newest_str = (
            newest.isoformat()
            if isinstance(newest, date)
            else (newest or today.isoformat())
        )
        data = self._request(
            "GET",
            f"athlete/{self.athlete_id}/activities",
            params={"oldest": oldest_str, "newest": newest_str},
        )
        return data if isinstance(data, list) else []

    def get_activity(self, activity_id: str) -> dict[str, Any]:
        """Fetch details of a single completed activity by ID."""
        return self._request("GET", f"activity/{activity_id}")

    def update_activity(
        self, activity_id: str, updates: dict[str, Any]
    ) -> dict[str, Any]:
        """Update an existing completed activity (e.g., enrich a Suunto strength session with description, RPE, feel, load)."""
        return self._request("PUT", f"activity/{activity_id}", json_body=updates)

    def get_events(
        self,
        oldest: str | date | None = None,
        newest: str | date | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch planned workouts and calendar notes in [oldest, newest] range."""
        today = date.today()
        oldest_str = (
            oldest.isoformat()
            if isinstance(oldest, date)
            else (oldest or today.isoformat())
        )
        newest_str = (
            newest.isoformat()
            if isinstance(newest, date)
            else (newest or (today + timedelta(days=14)).isoformat())
        )
        data = self._request(
            "GET",
            f"athlete/{self.athlete_id}/events",
            params={"oldest": oldest_str, "newest": newest_str},
        )
        return data if isinstance(data, list) else []

    @staticmethod
    def build_event_payload(workout: PlannedWorkoutInput) -> dict[str, Any]:
        """Convert a PlannedWorkoutInput into an intervals.icu Event JSON object."""
        dt_str = workout.date.strip()
        if len(dt_str) == 10:
            # Date only YYYY-MM-DD -> intervals.icu expects YYYY-MM-DDTHH:MM:SS
            dt_str = f"{dt_str}T00:00:00"
        else:
            # Validate ISO format
            datetime.fromisoformat(dt_str)

        if workout.sport_type == "Note":
            payload: dict[str, Any] = {
                "category": "NOTE",
                "start_date_local": dt_str,
                "name": workout.name,
                "description": workout.description,
            }
        else:
            payload = {
                "category": "WORKOUT",
                "type": workout.sport_type,
                "start_date_local": dt_str,
                "name": workout.name,
                "description": workout.description,
            }
            if workout.moving_time_minutes is not None:
                payload["moving_time"] = int(workout.moving_time_minutes * 60)
            if workout.estimated_load is not None:
                payload["icu_training_load"] = int(workout.estimated_load)

        if workout.external_id:
            payload["external_id"] = workout.external_id
        else:
            # Deterministic external_id by date + sport_type + slugified name for safe upsert
            slug = re.sub(r"[^a-z0-9]+", "-", workout.name.lower()).strip("-")[:32]
            payload["external_id"] = f"kinesis-{dt_str[:10]}-{workout.sport_type.lower()}-{slug}"

        return payload

    def create_events_bulk(
        self,
        workouts: list[PlannedWorkoutInput],
        upsert: bool = True,
    ) -> list[dict[str, Any]]:
        """Upload multiple planned workouts/notes to the intervals.icu calendar."""
        payloads = [self.build_event_payload(w) for w in workouts]
        result = self._request(
            "POST",
            f"athlete/{self.athlete_id}/events/bulk",
            params={"upsert": "true" if upsert else "false"},
            json_body=payloads,
        )
        return result if isinstance(result, list) else [result]

    def delete_event(self, event_id: int | str) -> dict[str, Any]:
        """Delete a planned calendar event by ID."""
        return self._request("DELETE", f"athlete/{self.athlete_id}/events/{event_id}")
