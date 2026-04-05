# models.py
#
# Pydantic models are the contract between all parts of your system.
# The agent speaks Action. The env responds with Observation.
# The grader returns Reward. Nothing passes untyped data between files.

from pydantic import BaseModel, field_validator
from typing import Optional, Any

# ── What the agent sends to the environment ──────────────────────────────────
class Action(BaseModel):
    # The agent has exactly 4 tools it can call. No others.
    # "run_sql"        → execute a SQL query against the broken database
    # "inspect_schema" → see column names and types of a table
    # "check_logs"     → read the dbt/pipeline changelog
    # "diff_report"    → compare report output between two dates
    tool: str
    query: str                    # the SQL string, table name, or date range
    reasoning: Optional[str] = None  # agent explains WHY it's making this call

# ── What the environment sends back ──────────────────────────────────────────
class Observation(BaseModel):
    task_id: int       # 1=easy, 2=medium, 3=hard
    result: Any        # the tool's output — could be rows, schema, log text
    steps_taken: int   # how many tool calls so far this episode
    done: bool         # True = episode over (agent submitted answer or ran out of steps)
    max_steps: int = 10  # agent knows it has a budget

# ── The scoring breakdown (used inside Reward) ────────────────────────────────
class RewardBreakdown(BaseModel):
    diagnosis_correct: float  # did agent name the right root cause? max 0.30
    fix_valid: float          # does their fix query return correct output? max 0.25
    efficiency: float         # optimal_steps/steps_taken ratio, max 0.20
    reasoning_quality: float  # did agent explain its thinking? max 0.15

# ── What the grader returns after an episode ends ────────────────────────────
class Reward(BaseModel):
    score: float                         # final score 0.0–1.0
    breakdown: RewardBreakdown
    false_positive_penalty: float = 0.0 # penalty if agent touched the wrong tables

    @field_validator("score")
    @classmethod
    def clamp_score(cls, v):
        # Ensures score never goes above 1.0 or below 0.0 no matter what
        return round(max(0.0, min(1.0, v)), 4)

# ── The full state (what state() returns) ────────────────────────────────────
class EnvState(BaseModel):
    task_id: int
    steps_taken: int
    done: bool
    agent_actions: list   # full history of what the agent did this episode
    current_observation: Optional[Observation] = None