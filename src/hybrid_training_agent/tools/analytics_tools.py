from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from statistics import mean, stdev
from typing import Any

from langchain_core.tools import tool

from hybrid_training_agent.clients.intervals_client import (
    IntervalsIcuClient,
    IntervalsIcuError,
)


def _format_pace_sec_per_km(speed_m_per_s: float | None) -> str | None:
    if not speed_m_per_s or speed_m_per_s <= 0:
        return None
    sec_per_km = 1000.0 / speed_m_per_s
    mins = int(sec_per_km // 60)
    secs = int(round(sec_per_km % 60))
    if secs == 60:
        mins += 1
        secs = 0
    return f"{mins}:{secs:02d}/km"


def compute_hybrid_analytics(
    wellness_records: list[dict[str, Any]],
    activities: list[dict[str, Any]],
    days_back: int = 28,
) -> dict[str, Any]:
    """Pure analytical engine computing load, readiness, polarization, and concurrent interference metrics."""
    # 1. Wellness & Load Metrics (CTL, ATL, TSB, ACWR, HRV, RHR, Sleep)
    sorted_wellness = sorted(
        wellness_records, key=lambda w: str(w.get("id") or w.get("date") or "")
    )
    latest_w = sorted_wellness[-1] if sorted_wellness else {}
    ctl = float(latest_w.get("ctl") or 0.0)
    atl = float(latest_w.get("atl") or 0.0)
    tsb = round(ctl - atl, 1)
    ramp_rate = latest_w.get("rampRate")
    acwr = round(atl / ctl, 2) if ctl > 5.0 else None

    if acwr is None:
        acwr_status = "Insufficient chronic load baseline (CTL <= 5)"
    elif acwr < 0.8:
        acwr_status = "Under-training / Detraining risk (<0.8)"
    elif acwr <= 1.3:
        acwr_status = "Optimal progressive overload 'Sweet Spot' (0.8 - 1.3)"
    elif acwr <= 1.5:
        acwr_status = "Elevated fatigue / Functional overreaching caution (1.3 - 1.5)"
    else:
        acwr_status = "High injury & maladaptation risk (>1.5)"

    # HRV & RHR 7-day vs 28-day baseline
    hrv_all = [
        float(w["hrv"])
        for w in sorted_wellness
        if w.get("hrv") is not None and float(w["hrv"]) > 0
    ]
    hrv_7d = [
        float(w["hrv"])
        for w in sorted_wellness[-7:]
        if w.get("hrv") is not None and float(w["hrv"]) > 0
    ]
    rhr_all = [
        float(w["restingHR"])
        for w in sorted_wellness
        if w.get("restingHR") is not None and float(w["restingHR"]) > 0
    ]
    rhr_7d = [
        float(w["restingHR"])
        for w in sorted_wellness[-7:]
        if w.get("restingHR") is not None and float(w["restingHR"]) > 0
    ]
    sleep_7d_hours = [
        float(w["sleepSecs"]) / 3600.0
        for w in sorted_wellness[-7:]
        if w.get("sleepSecs") is not None and float(w["sleepSecs"]) > 0
    ]

    hrv_summary: dict[str, Any] = {
        "latest_hrv_rmssd": hrv_all[-1] if hrv_all else None,
        "rolling_7d_mean": round(mean(hrv_7d), 1) if hrv_7d else None,
        "baseline_mean": round(mean(hrv_all), 1) if hrv_all else None,
        "rolling_7d_cv_pct": (
            round((stdev(hrv_7d) / mean(hrv_7d)) * 100.0, 1)
            if len(hrv_7d) >= 3 and mean(hrv_7d) > 0
            else None
        ),
    }
    if hrv_summary["rolling_7d_mean"] and hrv_summary["baseline_mean"]:
        pct_diff = (
            (hrv_summary["rolling_7d_mean"] - hrv_summary["baseline_mean"])
            / hrv_summary["baseline_mean"]
        ) * 100.0
        hrv_summary["pct_vs_baseline"] = round(pct_diff, 1)
        if pct_diff < -7.5:
            hrv_summary["autonomic_status"] = (
                "Suppressed sympathetic/parasympathetic balance (>7.5% below baseline)"
            )
        elif pct_diff > 7.5:
            hrv_summary["autonomic_status"] = "High parasympathetic readiness"
        else:
            hrv_summary["autonomic_status"] = "Normal autonomic baseline"

    rhr_summary = {
        "latest_rhr_bpm": rhr_all[-1] if rhr_all else None,
        "rolling_7d_mean_bpm": round(mean(rhr_7d), 1) if rhr_7d else None,
        "baseline_mean_bpm": round(mean(rhr_all), 1) if rhr_all else None,
    }

    sleep_summary = {
        "rolling_7d_avg_hours": round(mean(sleep_7d_hours), 2) if sleep_7d_hours else None,
        "latest_soreness": latest_w.get("soreness"),
        "latest_fatigue": latest_w.get("fatigue"),
        "latest_stress": latest_w.get("stress"),
        "latest_mood": latest_w.get("mood"),
    }

    # 2. Activities Categorization (Running vs Strength vs Other)
    runs: list[dict[str, Any]] = []
    strength_sessions: list[dict[str, Any]] = []
    other_activities: list[dict[str, Any]] = []

    # Zone time accumulators for running (seconds in Z1-Z7)
    run_hr_zone_secs = [0.0] * 7
    run_pace_zone_secs = [0.0] * 7
    total_run_distance_m = 0.0
    total_run_time_s = 0.0
    total_run_load = 0.0
    total_strength_time_s = 0.0

    for act in activities:
        act_type = str(act.get("type") or "").strip()
        start_local = str(act.get("start_date_local") or "")
        act_date = start_local[:10] if len(start_local) >= 10 else ""
        moving_time_s = float(act.get("moving_time") or act.get("elapsed_time") or 0.0)
        duration_min = round(moving_time_s / 60.0, 1)
        load = float(act.get("icu_training_load") or 0.0)

        if act_type in ("Run", "TrailRun", "VirtualRun", "Treadmill"):
            dist_m = float(act.get("distance") or 0.0)
            total_run_distance_m += dist_m
            total_run_time_s += moving_time_s
            total_run_load += load

            hr_zones = act.get("icu_hr_zone_times") or []
            for idx, z_sec in enumerate(hr_zones[:7]):
                if z_sec:
                    run_hr_zone_secs[idx] += float(z_sec)

            pace_zones = act.get("icu_pace_zone_times") or []
            for idx, z_sec in enumerate(pace_zones[:7]):
                if z_sec:
                    run_pace_zone_secs[idx] += float(z_sec)

            runs.append(
                {
                    "id": act.get("id"),
                    "date": act_date,
                    "start_date_local": start_local,
                    "name": act.get("name"),
                    "distance_km": round(dist_m / 1000.0, 2),
                    "duration_min": duration_min,
                    "avg_hr": act.get("average_heartrate"),
                    "max_hr": act.get("max_heartrate"),
                    "avg_pace": _format_pace_sec_per_km(act.get("average_speed")),
                    "gap_pace": _format_pace_sec_per_km(act.get("gap")),
                    "load": load,
                    "rpe": act.get("icu_rpe"),
                    "decoupling_pct": act.get("decoupling"),
                }
            )
        elif act_type in ("WeightTraining", "Strength", "Crossfit", "Workout"):
            total_strength_time_s += moving_time_s
            strength_sessions.append(
                {
                    "id": act.get("id"),
                    "date": act_date,
                    "start_date_local": start_local,
                    "name": act.get("name") or "Strength Training",
                    "duration_min": duration_min,
                    "avg_hr": act.get("average_heartrate"),
                    "load": load,
                    "rpe": act.get("icu_rpe"),
                }
            )
        else:
            other_activities.append(
                {
                    "id": act.get("id"),
                    "type": act_type,
                    "date": act_date,
                    "name": act.get("name"),
                    "duration_min": duration_min,
                    "load": load,
                }
            )

    # 3. Running Intensity Distribution (Seiler 3-Zone Model from 5/7 zones)
    active_zone_secs = (
        run_pace_zone_secs if sum(run_pace_zone_secs) > 0 else run_hr_zone_secs
    )
    total_z_secs = sum(active_zone_secs)
    if total_z_secs > 0:
        low_pct = round(((active_zone_secs[0] + active_zone_secs[1]) / total_z_secs) * 100.0, 1)
        mod_pct = round((active_zone_secs[2] / total_z_secs) * 100.0, 1)
        high_pct = round((sum(active_zone_secs[3:]) / total_z_secs) * 100.0, 1)
    else:
        low_pct, mod_pct, high_pct = 0.0, 0.0, 0.0

    polarization_index: float | None = None
    if low_pct > 0 and mod_pct > 0 and high_pct > 0:
        polarization_index = round(
            math.log10((low_pct / 100.0) / ((mod_pct / 100.0) * (high_pct / 100.0))),
            2,
        )

    # 4. Concurrent Interference Detection (Running vs Strength spacing)
    interference_warnings: list[str] = []
    for s_act in strength_sessions:
        s_dt_str = s_act.get("start_date_local") or ""
        s_date = s_act.get("date") or ""
        for r_act in runs:
            r_dt_str = r_act.get("start_date_local") or ""
            r_date = r_act.get("date") or ""
            if s_date and s_date == r_date:
                hours_apart: float | None = None
                try:
                    s_dt = datetime.fromisoformat(s_dt_str.replace("Z", "+00:00"))
                    r_dt = datetime.fromisoformat(r_dt_str.replace("Z", "+00:00"))
                    hours_apart = round(abs((r_dt - s_dt).total_seconds()) / 3600.0, 1)
                except Exception:
                    pass
                if hours_apart is not None and hours_apart < 6.0:
                    interference_warnings.append(
                        f"Same-day Interference Alert ({s_date}): '{s_act['name']}' and Run '{r_act['name']}' "
                        f"were performed only {hours_apart}h apart (<6h threshold per Robineau et al., 2016)."
                    )
                else:
                    interference_warnings.append(
                        f"Double-Day Concurrent Session ({s_date}): '{s_act['name']}' and Run '{r_act['name']}' "
                        f"({f'{hours_apart}h apart' if hours_apart is not None else 'same day'}). Ensure adequate carbohydrate/protein refueling."
                    )

    weeks_factor = max(days_back / 7.0, 1.0)
    return {
        "analysis_window_days": days_back,
        "load_and_form": {
            "ctl_fitness": round(ctl, 1),
            "atl_fatigue": round(atl, 1),
            "tsb_form": tsb,
            "ramp_rate": ramp_rate,
            "acwr_ratio": acwr,
            "acwr_interpretation": acwr_status,
        },
        "recovery_and_wellness": {
            "hrv": hrv_summary,
            "resting_hr": rhr_summary,
            "sleep_and_subjective": sleep_summary,
        },
        "running_summary": {
            "total_sessions": len(runs),
            "sessions_per_week": round(len(runs) / weeks_factor, 1),
            "total_distance_km": round(total_run_distance_m / 1000.0, 1),
            "weekly_avg_distance_km": round((total_run_distance_m / 1000.0) / weeks_factor, 1),
            "total_duration_hours": round(total_run_time_s / 3600.0, 2),
            "intensity_distribution_3_zone": {
                "zone_1_2_low_pct": low_pct,
                "zone_3_moderate_grey_zone_pct": mod_pct,
                "zone_4_5_high_pct": high_pct,
                "polarization_index": polarization_index,
                "target_recommendation": "Aim for ~75-80% Low (Z1-Z2), <5-10% Moderate (Z3), 15-20% High (Z4-Z5) per Seiler (2010).",
            },
            "recent_runs": runs[-10:],
        },
        "strength_summary": {
            "total_sessions": len(strength_sessions),
            "sessions_per_week": round(len(strength_sessions) / weeks_factor, 1),
            "total_duration_hours": round(total_strength_time_s / 3600.0, 2),
            "strength_sessions": strength_sessions,
        },
        "concurrent_interference_audit": {
            "warnings_count": len(interference_warnings),
            "warnings": interference_warnings,
        },
        # Compact per-activity rows (ascending). The analyst groups these by whatever period
        # or field the user asks for (weekly, monthly, per sport, ...) instead of the tool
        # hard-coding any single grouping.
        "activity_log": compact_activity_log(activities),
    }


def compact_activity_log(
    activities: list[dict[str, Any]], limit: int = 750
) -> list[dict[str, Any]]:
    """Reduce raw activities to compact, ascending rows suitable for regrouping."""
    rows: list[dict[str, Any]] = []
    for act in sorted(activities, key=lambda a: str(a.get("start_date_local") or "")):
        start = str(act.get("start_date_local") or "")
        moving_time_s = float(act.get("moving_time") or act.get("elapsed_time") or 0.0)
        distance_m = act.get("distance")
        rows.append(
            {
                "date": start[:10],
                "type": str(act.get("type") or ""),
                "name": act.get("name"),
                "distance_km": round(float(distance_m) / 1000.0, 2) if distance_m else None,
                "duration_min": round(moving_time_s / 60.0, 1) if moving_time_s else None,
                "load": act.get("icu_training_load"),
            }
        )
    if limit and len(rows) > limit:
        return rows[-limit:]
    return rows


@tool
def fetch_athlete_profile_and_zones() -> dict[str, Any]:
    """Fetch the athlete's intervals.icu profile, running threshold pace, LTHR, max HR, and training zones."""
    client = IntervalsIcuClient()
    try:
        profile = client.get_athlete_profile()
    except IntervalsIcuError as exc:
        return {"error": str(exc)}

    sport_settings = profile.get("sportSettings") or []
    run_settings: dict[str, Any] = {}
    for s in sport_settings:
        types = s.get("types") or []
        if "Run" in types:
            run_settings = {
                "types": types,
                "lthr_bpm": s.get("lthr"),
                "max_hr_bpm": s.get("max_hr"),
                "threshold_pace_m_per_s": s.get("threshold_pace"),
                "threshold_pace_per_km": _format_pace_sec_per_km(s.get("threshold_pace")),
                "hr_zones": s.get("hr_zones"),
                "pace_zones": s.get("pace_zones"),
                "pace_zone_names": s.get("pace_zone_names"),
            }
            break

    return {
        "id": profile.get("id"),
        "name": profile.get("name"),
        "weight_kg": profile.get("icu_weight"),
        "resting_hr_bpm": profile.get("icu_resting_hr"),
        "run_sport_settings": run_settings,
    }


@tool
def analyze_training_and_wellness(days_back: int = 28) -> dict[str, Any]:
    """Fetch past training activities and wellness data from intervals.icu and run a full hybrid training efficiency analysis.

    Analyses:
    1. Fitness (CTL), Fatigue (ATL), Form (TSB), and Acute:Chronic Workload Ratio (ACWR).
    2. Autonomic recovery: 7-day rolling HRV vs baseline, HRV CV%, Resting HR trends, and Sleep duration.
    3. Running volume, pace/HR zone distribution, and Polarization Index (80/20 rule check).
    4. Strength sessions (duration, frequency, HR) and concurrent interference spacing (<6h rule).
    """
    client = IntervalsIcuClient()
    today = date.today()
    oldest = (today - timedelta(days=days_back)).isoformat()
    newest = today.isoformat()

    try:
        wellness = client.get_wellness(oldest=oldest, newest=newest)
        activities = client.get_activities(oldest=oldest, newest=newest)
    except IntervalsIcuError as exc:
        return {"error": str(exc)}

    report = compute_hybrid_analytics(
        wellness_records=wellness,
        activities=activities,
        days_back=days_back,
    )
    report["window"] = {"oldest": oldest, "newest": newest, "days": days_back}
    return report


ANALYTICS_TOOLS = [
    fetch_athlete_profile_and_zones,
    analyze_training_and_wellness,
]


# ---------------------------------------------------------------------------
# Performance benchmarking & session-execution analysis (read-only)
# ---------------------------------------------------------------------------
def _normalize_streams(streams: Any) -> dict[str, list[Any]]:
    """Normalize intervals.icu stream payloads into `{stream_type: samples}`."""
    normalized: dict[str, list[Any]] = {}
    if isinstance(streams, dict):
        popped = streams.get("streams")
        if isinstance(popped, list):
            streams = popped
        else:
            for key, value in streams.items():
                if isinstance(value, list):
                    normalized[str(key)] = value
            return normalized
    if isinstance(streams, list):
        for item in streams:
            if not isinstance(item, dict):
                continue
            stream_type = item.get("type") or item.get("name")
            data = item.get("data")
            if stream_type and isinstance(data, list):
                normalized[str(stream_type)] = data
    return normalized


def _numeric_summary(values: list[Any]) -> dict[str, float] | None:
    numbers = [float(v) for v in values if isinstance(v, (int, float))]
    if not numbers:
        return None
    return {
        "count": len(numbers),
        "min": round(min(numbers), 2),
        "max": round(max(numbers), 2),
        "avg": round(sum(numbers) / len(numbers), 2),
    }


def _half_split_summary(values: list[Any]) -> dict[str, Any] | None:
    numbers = [float(v) for v in values if isinstance(v, (int, float))]
    if len(numbers) < 4:
        return None
    mid = len(numbers) // 2
    first = sum(numbers[:mid]) / mid
    second = sum(numbers[mid:]) / (len(numbers) - mid)
    drift = ((second - first) / first * 100.0) if first else None
    return {
        "first_half_avg": round(first, 2),
        "second_half_avg": round(second, 2),
        "drift_pct": round(drift, 1) if drift is not None else None,
    }


def _summarize_streams(streams: Any) -> dict[str, Any]:
    normalized = _normalize_streams(streams)
    summary: dict[str, Any] = {"available_streams": sorted(normalized)}
    samples = max((len(v) for v in normalized.values()), default=0)
    summary["samples"] = samples

    for stream_type in ("heartrate", "velocity_smooth", "watts", "cadence", "altitude"):
        stats = _numeric_summary(normalized.get(stream_type, []))
        if stats:
            summary[stream_type] = stats
            split = _half_split_summary(normalized.get(stream_type, []))
            if split:
                summary[stream_type]["half_split"] = split

    speed_stats = summary.get("velocity_smooth")
    if speed_stats:
        summary["pace_sec_per_km"] = {
            "avg": _format_pace_sec_per_km(speed_stats["avg"]),
            "best": _format_pace_sec_per_km(speed_stats["max"]),
        }
    return summary


def _summarize_intervals(intervals: Any) -> list[dict[str, Any]]:
    if isinstance(intervals, dict):
        intervals = (
            intervals.get("icu_intervals")
            or intervals.get("intervals")
            or intervals.get("data")
            or []
        )
    if not isinstance(intervals, list):
        return []

    summarized: list[dict[str, Any]] = []
    for interval in intervals:
        if not isinstance(interval, dict):
            continue
        summarized.append(
            {
                "id": interval.get("id"),
                "type": interval.get("type"),
                "start_index": interval.get("start_index"),
                "end_index": interval.get("end_index"),
                "duration_s": interval.get("moving_time") or interval.get("elapsed_time"),
                "distance_m": interval.get("distance"),
                "avg_hr": interval.get("average_heartrate"),
                "max_hr": interval.get("max_heartrate"),
                "avg_pace": _format_pace_sec_per_km(interval.get("average_speed")),
                "avg_watts": interval.get("average_watts"),
                "intensity_pct": interval.get("intensity"),
                "training_load": interval.get("training_load"),
                "zone": interval.get("zone"),
            }
        )
    return summarized


@tool
def get_performance_curves(sport: str = "Run", periods: str = "42d,1y") -> dict[str, Any]:
    """Fetch the athlete's best pace and heart-rate curves for benchmarking.

    Use this to calibrate paces/thresholds and estimate race potential. `periods` is a
    comma separated list of windows: e.g. '42d', '1y', '2y', 'all', 's0' (current season).
    """
    client = IntervalsIcuClient()
    try:
        pace = client.get_pace_curves(sport=sport, curves=periods)
        hr = client.get_hr_curves(sport=sport, curves=periods)
    except IntervalsIcuError as exc:
        return {"error": str(exc)}
    return {"sport": sport, "periods": periods, "pace_curves": pace, "hr_curves": hr}


@tool
def search_completed_activities(query: str, limit: int = 20) -> dict[str, Any]:
    """Search completed activities by name, or exact tag when the query starts with '#'.

    Useful for finding previous sessions (e.g. 'VO2max', 'long run', '#threshold').
    """
    client = IntervalsIcuClient()
    try:
        results = client.search_activities(query=query, limit=limit)
    except IntervalsIcuError as exc:
        return {"error": str(exc)}
    return {"query": query, "count": len(results) if isinstance(results, list) else None, "activities": results}


@tool
def search_similar_interval_sessions(
    min_seconds: int,
    max_seconds: int,
    min_intensity_pct: float,
    max_intensity_pct: float,
    sport: str = "Run",
    min_reps: int | None = None,
    max_reps: int | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Find completed activities containing intervals that match a duration/intensity band.

    Intensity is a percentage of the athlete's threshold (e.g. 98-102 for threshold pace).
    Use this to answer "when did I last run 5x1km at this pace?" or to compare blocks.
    """
    client = IntervalsIcuClient()
    try:
        results = client.search_interval_activities(
            min_secs=min_seconds,
            max_secs=max_seconds,
            min_intensity=min_intensity_pct,
            max_intensity=max_intensity_pct,
            sport=sport,
            min_reps=min_reps,
            max_reps=max_reps,
            limit=limit,
        )
    except IntervalsIcuError as exc:
        return {"error": str(exc)}
    return {
        "criteria": {
            "duration_s": [min_seconds, max_seconds],
            "intensity_pct": [min_intensity_pct, max_intensity_pct],
            "sport": sport,
            "reps": [min_reps, max_reps],
        },
        "count": len(results) if isinstance(results, list) else None,
        "activities": results,
    }


@tool
def analyze_activity_execution(
    activity_id: str,
    include_streams: bool = True,
    stream_types: str = "time,heartrate,velocity_smooth,watts,cadence",
) -> dict[str, Any]:
    """Review how a completed session was executed.

    Returns the detected intervals/reps (duration, distance, avg pace/HR/power, intensity)
    plus a compact stream summary (averages, bests, and first/second-half drift) so you can
    check whether the athlete hit the prescribed targets. Pass the activity id from the
    pre-fetched recent activities.
    """
    client = IntervalsIcuClient()
    try:
        intervals = client.get_activity_intervals(activity_id)
        result: dict[str, Any] = {
            "activity_id": activity_id,
            "intervals": _summarize_intervals(intervals),
        }
        if include_streams:
            streams = client.get_activity_streams(activity_id, types=stream_types)
            result["stream_summary"] = _summarize_streams(streams)
        return result
    except IntervalsIcuError as exc:
        return {"error": str(exc)}


@tool
def get_activity_interval_stats(
    activity_id: str, start_index: int, end_index: int
) -> dict[str, Any]:
    """Compute interval-like stats for an arbitrary slice of an activity's streams.

    Use `start_index`/`end_index` from `analyze_activity_execution` output.
    """
    client = IntervalsIcuClient()
    try:
        return client.get_interval_stats(
            activity_id, start_index=start_index, end_index=end_index
        )
    except IntervalsIcuError as exc:
        return {"error": str(exc)}


@tool
def get_activity_best_efforts(
    activity_id: str,
    duration_seconds: int | None = None,
    distance_meters: float | None = None,
    stream: str = "velocity_smooth",
    count: int = 5,
) -> dict[str, Any]:
    """Find the best efforts inside a completed activity for a duration or distance."""
    if duration_seconds is None and distance_meters is None:
        return {"error": "Provide duration_seconds or distance_meters."}
    client = IntervalsIcuClient()
    try:
        return client.get_activity_best_efforts(
            activity_id,
            stream=stream,
            duration=duration_seconds,
            distance=distance_meters,
            count=count,
        )
    except IntervalsIcuError as exc:
        return {"error": str(exc)}


PERFORMANCE_ANALYSIS_TOOLS = [
    get_performance_curves,
    search_completed_activities,
    search_similar_interval_sessions,
    analyze_activity_execution,
    get_activity_interval_stats,
    get_activity_best_efforts,
]
