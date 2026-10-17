from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any

import httpx

from hybrid_training_agent.clients.cache import TTLCache
from hybrid_training_agent.clients.http_client import get_shared_http_client
from hybrid_training_agent.config import Settings, get_settings
from hybrid_training_agent.models import PlannedWorkoutInput

# Shared read cache: sub-agents and multi-turn chat requests repeatedly hit the same
# profile/wellness/activity endpoints within seconds of each other.
_intervals_cache = TTLCache()


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
        *,
        use_cache: bool | None = None,
    ) -> Any:
        auth = self._require_credentials()
        method_upper = method.upper()
        url = f"{self.base_url}/{path.lstrip('/')}"
        cache_key = (
            method_upper,
            url,
            tuple(sorted((str(k), str(v)) for k, v in (params or {}).items())),
        )
        cacheable = method_upper == "GET" if use_cache is None else use_cache
        if cacheable:
            cached = _intervals_cache.get(cache_key)
            if cached is not None:
                return cached
        try:
            response = get_shared_http_client(self.timeout).request(
                method=method,
                url=url,
                params=params,
                json=json_body,
                headers={"Accept": "application/json"},
                auth=auth,
            )
            response.raise_for_status()
            if not response.content:
                data: Any = {}
            else:
                data = response.json()
            if method_upper == "GET":
                if cacheable:
                    _intervals_cache.set(cache_key, data)
            else:
                # Any write invalidates cached reads so uploads/deletes are reflected immediately.
                _intervals_cache.clear()
            return data
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

        # 'Note' events are calendar notes. Strength sessions are uploaded as notes too:
        # intervals.icu only builds SuuntoPlus Guides for WORKOUT events, and Suunto rejects
        # a guide with no steps. Strength sessions are plain readable text (no step lines),
        # so pushing them as workouts fails with "Suunto upload failed" every time.
        if workout.sport_type in ("Note", "WeightTraining"):
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

    def get_event(self, event_id: int | str) -> dict[str, Any]:
        """Fetch a single planned calendar event by ID."""
        return self._request("GET", f"athlete/{self.athlete_id}/events/{event_id}")

    def update_event(self, event_id: int | str, updates: dict[str, Any]) -> dict[str, Any]:
        """Update a single planned calendar event (partial update)."""
        return self._request(
            "PUT", f"athlete/{self.athlete_id}/events/{event_id}", json_body=updates
        )

    def apply_plan(
        self,
        folder_id: int,
        start_date_local: str,
        extra_workouts: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Apply a workout folder/plan from the library to the athlete's calendar."""
        body: dict[str, Any] = {
            "folder_id": int(folder_id),
            "start_date_local": start_date_local,
        }
        if extra_workouts:
            body["extra_workouts"] = extra_workouts
        return self._request(
            "POST", f"athlete/{self.athlete_id}/events/apply-plan", json_body=body
        )

    # ------------------------------------------------------------------
    # Performance benchmarking reads
    # ------------------------------------------------------------------
    def get_pace_curves(
        self,
        sport: str = "Run",
        curves: str = "42d,1y",
        include_ranks: bool = False,
        gap: bool = False,
    ) -> Any:
        """Best pace curves for the athlete (e.g. 5s..2h best average pace)."""
        params: dict[str, Any] = {"type": sport, "curves": curves}
        if include_ranks:
            params["includeRanks"] = "true"
        if gap:
            params["gap"] = "true"
        return self._request(
            "GET", f"athlete/{self.athlete_id}/pace-curves.json", params=params
        )

    def get_hr_curves(self, sport: str = "Run", curves: str = "42d,1y") -> Any:
        """Best heart-rate curves for the athlete."""
        return self._request(
            "GET",
            f"athlete/{self.athlete_id}/hr-curves.json",
            params={"type": sport, "curves": curves},
        )

    def search_activities(self, query: str, limit: int = 20) -> Any:
        """Search completed activities by name, or exact tag when the query starts with '#'."""
        return self._request(
            "GET",
            f"athlete/{self.athlete_id}/activities/search",
            params={"q": query, "limit": limit},
        )

    def search_interval_activities(
        self,
        min_secs: int,
        max_secs: int,
        min_intensity: float,
        max_intensity: float,
        sport: str | None = None,
        min_reps: int | None = None,
        max_reps: int | None = None,
        limit: int = 10,
    ) -> Any:
        """Find activities containing intervals matching a duration and intensity band."""
        params: dict[str, Any] = {
            "minSecs": min_secs,
            "maxSecs": max_secs,
            "minIntensity": min_intensity,
            "maxIntensity": max_intensity,
            "limit": limit,
        }
        if sport:
            params["type"] = sport
        if min_reps is not None:
            params["minReps"] = min_reps
        if max_reps is not None:
            params["maxReps"] = max_reps
        return self._request(
            "GET", f"athlete/{self.athlete_id}/activities/interval-search", params=params
        )

    def get_activity_streams(
        self,
        activity_id: int | str,
        types: str | None = None,
        include_defaults: bool = False,
    ) -> Any:
        """Fetch per-sample activity streams (time, HR, pace/speed, power, cadence, distance)."""
        params: dict[str, Any] = {"includeDefaults": "true" if include_defaults else "false"}
        if types:
            params["types"] = types
        return self._request(
            "GET",
            f"activity/{activity_id}/streams.json",
            params=params,
            use_cache=False,
        )

    def get_activity_intervals(self, activity_id: int | str) -> Any:
        """Fetch detected/manual intervals (reps) for a completed activity."""
        return self._request("GET", f"activity/{activity_id}/intervals")

    def get_interval_stats(
        self, activity_id: int | str, start_index: int, end_index: int
    ) -> Any:
        """Fetch interval-like stats for a slice of an activity's streams."""
        return self._request(
            "GET",
            f"activity/{activity_id}/interval-stats",
            params={"start_index": start_index, "end_index": end_index},
        )

    def get_activity_best_efforts(
        self,
        activity_id: int | str,
        stream: str = "velocity_smooth",
        duration: int | None = None,
        distance: float | None = None,
        count: int = 5,
    ) -> Any:
        """Find best efforts within a completed activity for a duration or distance."""
        params: dict[str, Any] = {"stream": stream, "count": count}
        if duration is not None:
            params["duration"] = duration
        if distance is not None:
            params["distance"] = distance
        return self._request(
            "GET", f"activity/{activity_id}/best-efforts", params=params, use_cache=False
        )

    # ------------------------------------------------------------------
    # Workout library / plans
    # ------------------------------------------------------------------
    def get_workouts(self) -> list[dict[str, Any]]:
        """List all workouts in the athlete's workout library."""
        data = self._request("GET", f"athlete/{self.athlete_id}/workouts")
        return data if isinstance(data, list) else []

    def create_workout(self, workout: dict[str, Any]) -> dict[str, Any]:
        """Create a workout in the athlete's library (optionally inside a folder/plan)."""
        return self._request(
            "POST", f"athlete/{self.athlete_id}/workouts", json_body=workout
        )

    def get_folders(self) -> list[dict[str, Any]]:
        """List the athlete's workout folders and plans."""
        data = self._request("GET", f"athlete/{self.athlete_id}/folders")
        return data if isinstance(data, list) else []

    def get_training_plan(self) -> Any:
        """Get the athlete's currently applied training plan."""
        return self._request("GET", f"athlete/{self.athlete_id}/training-plan")
