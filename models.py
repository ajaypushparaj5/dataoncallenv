"""Data models for DataOnCallEnv.
Includes Action, Observation, Reward, and EnvState models.
"""

from pydantic import BaseModel, field_validator
from typing import Optional, Any


class Action(BaseModel):
    """Encapsulates an agent's tool call."""
    # Tools the agent can call:
    # "run_sql"        → execute a SQL query against the broken database
    # "inspect_schema" → see column names and types of a table
    # "check_logs"     → read the dbt/pipeline changelog
    # "check_airflow"  → read Airflow DAG run history
    # "diff_report"    → compare report output between two dates
    # "list_tables"    → discover available tables
    # "submit"         → submit root cause + fix. Ends episode.
    tool: str
    query: str                    # the SQL string, table name, or date range
    reasoning: Optional[str] = None  # agent explains WHY it's making this call


class Observation(BaseModel):
    """Encapsulates the environment's response to an action."""
    task_id: int       # 1=easy, 2=medium, 3=hard
    result: Any        # the tool's output — could be rows, schema, log text
    steps_taken: int   # how many tool calls so far this episode
    done: bool         # True = episode over (agent submitted answer or ran out of steps)
    max_steps: int = 15  # agent knows it has a step budget
    cost_spent: float = 0.0       # cumulative query cost this episode
    budget_remaining: float = 20.0  # remaining cost budget


class RewardBreakdown(BaseModel):
    """Detailed scoring components."""
    diagnosis_correct: float    # did agent name the right root cause? max 0.25
    fix_valid: float            # does their fix query return correct output? max 0.25
    efficiency: float           # combined step + cost efficiency, max 0.15
    reasoning_quality: float    # did agent explain its thinking? max 0.10
    investigation_quality: float = 0.0  # logical debugging sequence? max 0.10


class Reward(BaseModel):
    """Final score and breakdown for an episode."""
    score: float                         # final score 0.0–1.0
    breakdown: RewardBreakdown
    false_positive_penalty: float = 0.0  # penalty for irrelevant/duplicate queries

    @field_validator("score")
    @classmethod
    def clamp_score(cls, v):
        # Ensures score never goes above 1.0 or below 0.0 no matter what
        return round(max(0.0, min(1.0, v)), 4)


class EnvState(BaseModel):
    """Full internal state of the environment."""
    task_id: int
    steps_taken: int
    done: bool
    agent_actions: list   # full history of what the agent did this episode
    current_observation: Optional[Observation] = None
    discovered_tables: list = []  # tables the agent has discovered via list_tables
    cost_spent: float = 0.0
    budget_remaining: float = 20.0