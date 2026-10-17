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
│ • Strength & Run      │  │ • Verifiable PMID &  │  │ • Strength: written     │
│   Spacing Audit       │  │   DOI Citations      │  │   in intervals.icu      │
└───────────────────────┘  └──────────────────────┘  └─────────────────────────┘
```

### How Workouts Are Planned & Synced
- **Running Workouts (`Run`):** Written in **Intervals.icu Workout Builder syntax** (`- 15m 65-75% Pace`, `5x`, etc.) so `intervals.icu` automatically syncs structured intervals to your **Suunto** watch.
- **Strength Workouts (`WeightTraining`):** Written directly as readable workout descriptions on your **intervals.icu** calendar (exercises, sets, reps, rest)—no Suunto watch sync or special manual logging needed.

### Latency Optimizations
`ask` and `chat` are tuned to minimize network trips and model round trips:
- **Deterministic pre-fetch:** the analyst's profile + training/wellness analysis, the researcher's curated evidence, and the planner's upcoming calendar are fetched up front and injected into the sub-agent context, so an agent answers in one model call instead of spending a turn choosing tools. The analyst honors an explicit window (e.g. `last 60 days`).
- **Concurrent I/O:** independent tool calls (and pre-fetches) run in a thread pool instead of sequentially.
- **Connection reuse + read cache:** one pooled `httpx` client serves both intervals.icu and PubMed, and read responses are cached in-process for `KINESIS_CACHE_TTL_SECONDS` (default `60`, `0` disables). Any write (upload/delete) invalidates cached reads.
- **Lean tool payloads:** the planner binds only `validate_and_preview_training_plan` and `upload_training_plan_to_intervals` (plus the delete tool for removal requests); the literature sub-agent runs only when evidence is requested.


---

## Quick Start

### 1. Configure `.env`
Copy `.env.example` to `.env` in the project root and add your API keys:
```powershell
Copy-Item .env.example .env
```
- **LLM API Key**: Set `OPENCODE_GO_API_KEY` (or `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_API_KEY`).
- **Intervals.icu API Key**: Set `INTERVALS_ICU_API_KEY` (found in `intervals.icu` → *Settings* → *Developer Settings* → *API Key*). Leave `INTERVALS_ICU_ATHLETE_ID=0` (which automatically resolves your authenticated athlete profile).

### 2. CLI Commands

Run from the project root:

```powershell
# 1. Check configuration and test live intervals.icu connection
uv run kinesis status

# 2. Launch interactive multi-turn coaching chat
uv run kinesis chat

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
You can expose all 8 tools (`intervals.icu` analytics, PubMed search, and workout plan validator/uploader) as a standard Model Context Protocol (MCP) server:
```powershell
uv run kinesis mcp
```

### Reusable Agent Skill
A dedicated skill is included at `skills/hybrid-training-coach/SKILL.md` containing the citation standards, concurrent training recovery windows, and `intervals.icu` workout step syntax for Suunto watch synchronization.
