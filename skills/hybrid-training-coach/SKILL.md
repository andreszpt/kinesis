---
name: hybrid-training-coach
description: Evidence-based Hybrid Training (Running + Strength) methodology, PubMed citation standards, and Intervals.icu Workout Builder syntax.
---

# Kinesis — Hybrid Training Coach Skill (Running + Strength)

Use this skill when analyzing training/wellness data, researching sports science literature, or building hybrid Running + Strength training plans for `intervals.icu`.

## 1. Core Scientific Principles & Required Citations

Always provide verifiable peer-reviewed citations (**Authors, Year, Journal, PMID, DOI**):

1. **Concurrent Interference & Session Spacing**:
   - *Schumann et al. (2022)*, Sports Medicine (`PMID: 34757594`, `DOI: 10.1007/s40279-021-01587-7`): Concurrent training does not blunt maximal strength or hypertrophy when total volume is managed, but explosive strength / Rate of Force Development (RFD) is attenuated when sessions occur within `<3h`.
   - *Robineau et al. (2016)*, J Strength Cond Res (`PMID: 25546444`, `DOI: 10.1519/JSC.0000000000000798`): Enforce at least **6 hours** between same-day strength and running sessions, and **24 hours** before high-intensity intervals after heavy lower-body lifting.
   - *Wilson et al. (2012)*, J Strength Cond Res (`PMID: 22002517`): Running induces higher eccentric muscle damage than cycling; avoid high-rep lower-body failure before key runs.

2. **Strength Training for Running Economy**:
   - *Blagrove et al. (2018)*, Sports Medicine (`PMID: 29249083`, `DOI: 10.1007/s40279-017-0835-7`): 2 sessions/week of heavy neural resistance training (`3–5 sets × 3–6 reps @ ≥80% 1RM / 1–3 RIR`) + plyometrics improves running economy by **2–8%** and time-trial performance by **3–5%** without unwanted mass gain.
   - *Rønnestad & Mujika (2014)*, Scand J Med Sci Sports (`PMID: 23914932`, `DOI: 10.1111/sms.12104`).

3. **Running Intensity Distribution (80/20 Polarization)**:
   - *Seiler (2010)*, IJSPP (`PMID: 20861519`, `DOI: 10.1123/ijspp.5.3.276`) & *Stöggl & Sperlich (2014)*, Front Physiol (`PMID: 24550842`): Keep ~80% of running sessions below LT1 (Zone 1–2) and ~20% at LT2/VO2max (Zone 4–5).

4. **HRV Readiness**:
   - *Plews et al. (2013)*, Sports Medicine (`PMID: 23821469`): Monitor 7-day rolling RMSSD mean and coefficient of variation (CV%) relative to 28-day baseline.

## 2. Intervals.icu Workout Formatting

- **Warm-up & cool-down are opt-in**: Do NOT add warm-up or cool-down steps by default — only include them when the athlete explicitly asks for a warm-up and/or cool-down. The main work set is the default for both running and strength.
- **Running Workouts (`Run`) — Synced to Suunto**:
  Format `description` using `intervals.icu` step lines starting with `- ` so `intervals.icu` compiles the workout and syncs it to the athlete's Suunto watch:
  ```text
  Main Set 5x
  - 1km 98-102% Pace
  - 2m 60-70% Pace
  ```
  Add `- 15m 65-75% Pace` style warm-up/cool-down steps only when explicitly requested.
- **Strength Workouts (`WeightTraining`) — Written in Intervals.icu**:
  Write the workout directly in `intervals.icu` event `description` as clean, readable text (exercises, sets, reps, rest). No special watch sync or post-session logging is required, and no warm-up/cool-down sections unless explicitly requested.
