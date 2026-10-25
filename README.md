# ⚡ Kinesis — Hybrid Training AI Coach (Running + Strength)

**Kinesis** is a **LangGraph Multi-Agent** system designed to research peer-reviewed sports science literature, analyze past training and health data from **intervals.icu**, and generate/upload evidence-based hybrid **Running + Strength** training plans.

---

## Architecture: Supervisor + 3 Specialized Sub-Agents

```text
                       ┌──────────────────────────┐
                       │     Supervisor Node      │
                       │ (Intent & State Router)  │
                       └────┬────────┬────────┬───┘
                            │        │        │
        ┌───────────────────┘        │        └───────────────────┐
        ▼                            ▼                            ▼
┌───────────────────────┐  ┌──────────────────────┐  ┌─────────────────────────┐
│  1. Data & Wellness   │  │  2. Sports Science   │  │  3. Hybrid Coach &      │
│     Analyst Agent     │  │   Researcher Agent   │  │     Planner Agent       │
├───────────────────────┤  ├──────────────────────┤  ├─────────────────────────┤
│ • Intervals.icu API   │  │ • PubMed E-Utilities │  │ • Synthesizes Analyst + │
│ • CTL / ATL / TSB     │  │ • Europe PMC REST    │  │   Researcher findings   │
│ • ACWR & Polarization │  │ • Curated Hybrid     │  │ • Running: structured   │
│ • HRV (7d RMSSD/CV%)  │  │   Meta-Analyses      │  │   syntax for Suunto     │
│ • Strength & Run      │  │ • Verifiable PMID &  │  │ • Strength: structured  │
│   Spacing Audit       │  │   DOI Citations      │  │   workout (Suunto sync) │
│ • Pace/HR curves, PBs │  │                      │  │ • Workout library,      │
│ • Session execution   │  │                      │  │   plans, event edits    │
└───────────────────────┘  └──────────────────────┘  └─────────────────────────┘
```

### Routing: Three Tiers
The supervisor decides who runs, in three tiers:
1. **Tier 1 — deterministic heuristic** (`classify_intent`, zero latency). Used when intent is explicit: a plan keyword (*plan, schedule, program, mesocycle, periodize, prescribe, upload*) or an action verb (*create, move, save, delete, apply, …*) aimed at a workout/training noun; a data/quantity question (*how many, what's my, when did I, km, weekly, summary*); or an evidence request (*pubmed, citations, science, why*). This keeps the common cases instant and testable.
2. **Tier 2 — structured-output LLM supervisor**. Consulted **only** when Tier 1 finds no signal (e.g. short follow-ups like "sounds good, go ahead"). It returns one decision (`analyst` / `researcher` / `planner` / `FINISH`) via `with_structured_output(RouteDecision)`. Set `KINESIS_ROUTER_MODEL` to route with a cheaper model; if the call fails, the heuristic plan is used as a fallback.
3. **Tier 3 — per-step `Command` loop**. The supervisor is a real loop: it returns a LangGraph `Command(goto=…)` for **one hop at a time** (rather than a fixed analyst→researcher→planner list), re-deciding after each non-terminal sub-agent, so it can stop early, skip an agent, or follow up before finishing. The planner is terminal, so its final answer ends the turn without another router call.

Routing regressions are guarded by a labelled eval set (`tests/routing_eval_cases.json`) asserting the Tier-1 heuristic, plus unit tests for the Tier-2 router and its fallback.

### Performance Analysis (Analyst, read-only)
Pure data questions — e.g. `"how many kms did I run this week"`, `"what's my weekly volume"`, `"how did I execute my last run"` — route to the **analyst only** and never invoke the planner. Planning requires explicit intent (a plan keyword, or an action verb such as *create/move/save/delete* aimed at a workout/training noun).

Beyond the pre-fetched load/wellness snapshot, the analyst can drill into the athlete's data when the request is analysis-style (`analyze`, `compare`, `how did`, `PB`, `when did I last`, …):
- **`get_performance_curves`** — best pace/HR curves (`42d`, `1y`, `all`, season) for threshold calibration and race potential.
- **`analyze_activity_execution`** — detected intervals/reps (duration, distance, avg pace/HR/power, intensity) plus a compact stream summary (averages, bests, first/second-half drift).
- **`search_similar_interval_sessions`** / **`search_completed_activities`** — find comparable past sessions ("when did I last run 5×1km at this pace?").
- **`get_activity_interval_stats`** / **`get_activity_best_efforts`** — arbitrary-slice stats and best efforts (duration or distance).
- **Date-aware aggregation:** when the request references a period (`since 2025`, `last 3 months`, `this year`), the pre-fetched window covers exactly that period and includes a compact per-activity `activity_log`. The analyst groups those rows however you ask — weekly, monthly, per sport, cumulative — and reports exact sums. There is no per-period tool and no hard-coded grouping.

### Workout Library, Plans & Calendar Edits (Planner)
- **`get_workout_library`** / **`get_athlete_training_plan`** — inspect saved workouts, folders/plans and the applied plan.
- **`save_workouts_to_library`** — save reusable workouts (optionally into a folder/plan).
- **`apply_training_plan_folder`** — schedule a saved plan folder onto the calendar from a start date.
- **`update_calendar_event`** — move, reschedule, rename or edit a single calendar event.

### How Workouts Are Planned & Synced
- **Running Workouts (`Run`):** Written in **Intervals.icu Workout Builder syntax** (`- 15m 65-75% Pace`, `5x`, etc.) so `intervals.icu` automatically syncs structured intervals to your **Suunto** watch.
- **Strength Workouts (`WeightTraining`):** Uploaded as **structured workouts** in `intervals.icu` (readable exercise text for exercises, sets, reps, rest). Because `intervals.icu` only builds a SuuntoPlus Guide for `WORKOUT` events and Suunto rejects a guide with zero steps, a single duration-only **summary step** covering the session (e.g. `- 55m`) is appended automatically. The session therefore syncs to Suunto as a valid guide and never triggers the `Suunto upload failed` flag, while the planner's `estimated_load` is preserved.

### Confirmation Gate (writes & deletes)
Every mutating tool defaults to `confirm=false` and returns a `confirmation_required` preview instead of acting:
`upload_training_plan_to_intervals`, `delete_planned_event_from_intervals`, `update_calendar_event`, `save_workouts_to_library`, `apply_training_plan_folder`.

The planner must show that preview to you and ask for explicit confirmation, then re-call the tool with `confirm=true` only after you agree. The planner is instructed never to set `confirm=true` on its own. In `ask`, the built-in follow-up prompt lets you answer the confirmation; in `chat` you simply reply on the next turn. The same gate applies to the MCP server tools.

### Latency Optimizations
`ask` and `chat` are tuned to minimize network trips and model round trips:
- **Deterministic pre-fetch:** the analyst's profile + training/wellness analysis, the researcher's curated evidence, and the planner's upcoming calendar are fetched up front and injected into the sub-agent context, so an agent answers in one model call instead of spending a turn choosing tools. The analyst honors the period the request refers to (`last 60 days`, `since 2025`, `this year`) and includes a per-activity `activity_log` for on-the-fly grouping.
- **Concurrent I/O:** independent tool calls and pre-fetches (including wellness + activity reads) run in a thread pool instead of sequentially.
- **Connection reuse + read cache:** the main LLM client is reused across agents and chat turns, one pooled `httpx` client serves both intervals.icu and PubMed, and read responses are cached in-process for `KINESIS_CACHE_TTL_SECONDS` (default `60`, `0` disables). Any write (upload/delete) invalidates cached reads.
- **Lean tool payloads:** the planner normally binds only `validate_and_preview_training_plan` + `upload_training_plan_to_intervals`, adding delete/update/library tools solely when the request mentions them; the analyst's benchmarking tools are bound only for analysis-style requests; the literature sub-agent runs only when evidence is requested.



---

## Quick Start

### 1. Configure `.env`
Copy `.env.example` to `.env` in the project root and add your API keys:
```powershell
Copy-Item .env.example .env
```
- **LLM API Key**: Set `OPENCODE_GO_API_KEY` (or `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_API_KEY`).
- **Intervals.icu API Key**: Set `INTERVALS_ICU_API_KEY` (found in `intervals.icu` → *Settings* → *Developer Settings* → *API Key*). Leave `INTERVALS_ICU_ATHLETE_ID=0` (which automatically resolves your authenticated athlete profile).

### 2. TUI & CLI Commands

Run from the project root:

```powershell
# 1. Launch the full-screen interactive Textual TUI (Chat + Readiness Sidebar + Analytics/Calendar/Evidence Tabs)
uv run kinesis
# (or explicitly: uv run kinesis tui)

# 2. Check configuration and test live intervals.icu connection
uv run kinesis status

# 3. Launch classic inline multi-turn coaching chat
uv run kinesis chat
# In-chat quick commands: /plan [request] (plan one or more workouts), /analyze, /evidence,
# /status, /clear, /quit. Example: /plan 45m easy run tomorrow + 60m full body strength Friday

# 3. Ask a one-shot question or request a full training plan
# Plans are practical by default (analyst data check + planner). Add wording like
# "backed by scientific evidence" or "with citations" to run the literature sub-agent
# and get a PMID/DOI-backed rationale section.
uv run kinesis ask "Create a 60m full body strength training for tomorrow"
uv run kinesis ask "Analyze my last 28 days on intervals.icu and build a 7-day hybrid running + strength plan backed by scientific evidence"

# If the agents end their turn with a clarifying question (e.g. equipment, weekly
# frequency, or whether to upload), `ask` prompts you to reply and continues the run.
# This is skipped automatically when output is piped or stdin is not a terminal.
# The shortcut from option 2 below, `uv run kinesis chat`, remains the way to hold a
# longer multi-turn coaching conversation.

# 4. Run instant quantitative analysis from intervals.icu (add --ai for full LLM commentary)
uv run kinesis analyze --days 28

# 5. Search live PubMed / Europe PMC or browse curated hybrid training meta-analyses
uv run kinesis evidence --topic all
uv run kinesis evidence --query "concurrent training running economy heavy resistance"
```

---

## Using as an MCP Server & OpenCode Skill

### MCP Server
You can expose **20 tools** (intervals.icu analytics, performance benchmarking, activity execution analysis, workout library/plans, plan validator/uploader, and PubMed search) as a standard Model Context Protocol (MCP) server. Mutating tools (`upload`, `delete`, `update event`, `save to library`, `apply plan`) keep the same confirmation gate: call with `confirm=false` for a preview, then `confirm=true` after the user agrees.
```powershell
uv run kinesis mcp
```

### Reusable Agent Skill
A dedicated skill is included at `skills/hybrid-training-coach/SKILL.md` containing the citation standards, concurrent training recovery windows, and `intervals.icu` workout step syntax for Suunto watch synchronization.
