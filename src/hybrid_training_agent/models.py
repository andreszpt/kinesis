from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class ScientificCitation(BaseModel):
    """Peer-reviewed scientific reference supporting a training recommendation."""

    title: str = Field(..., description="Full paper title")
    authors: str = Field(..., description="Lead author(s), e.g., 'Schumann et al.'")
    year: int | str = Field(..., description="Publication year")
    journal: str = Field(default="", description="Journal name")
    pmid: str | None = Field(default=None, description="PubMed ID (PMID)")
    doi: str | None = Field(default=None, description="Digital Object Identifier (DOI)")
    study_type: str = Field(
        default="Peer-Reviewed Study",
        description="e.g., Meta-Analysis, Systematic Review, RCT, Cohort Study",
    )
    key_findings: str = Field(
        ..., description="Summary of quantitative findings and effect sizes"
    )
    practical_application: str = Field(
        ...,
        description="How this evidence directly informs the hybrid running + strength plan",
    )


class PlannedWorkoutInput(BaseModel):
    """A planned workout or calendar event to validate and upload to intervals.icu."""

    date: str = Field(
        ...,
        description="Workout date or local datetime in 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS' format",
    )
    name: str = Field(..., description="Short, descriptive workout title")
    sport_type: Literal["Run", "WeightTraining", "Note"] = Field(
        ...,
        description="'Run' for running workouts (synced to Suunto), 'WeightTraining' for strength workouts written in intervals.icu, or 'Note' for calendar notes",
    )
    description: str = Field(
        ...,
        description=(
            "For 'Run': MUST include valid intervals.icu workout builder steps (e.g. 'Warmup\\n- 10m 65-75% Pace\\n\\nMain Set 4x\\n- 4m 98-102% Pace\\n- 2m 60-70% Pace\\n\\nCooldown\\n- 10m 65-75% Pace') so it syncs to Suunto. "
            "For 'WeightTraining': Plain text / markdown description written directly in intervals.icu with exercises, sets, reps, and rest times."
        ),
    )
    moving_time_minutes: int | None = Field(
        default=None, description="Planned workout duration in minutes"
    )
    estimated_load: int | None = Field(
        default=None,
        description="Estimated icu_training_load (TSS equivalent) for CTL/ATL/TSB projection",
    )
    external_id: str | None = Field(
        default=None,
        description="Optional unique external_id for idempotent upserts on intervals.icu",
    )
