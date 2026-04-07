# api/app.py
# Production FastAPI server.
# The hackathon validator will hit /health, /reset, /step, /state.
# All must return proper JSON and correct HTTP status codes.

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Action, Observation, Reward, EnvState
from environment import DataOnCallEnv
from tasks import TASKS

app = FastAPI(
    title="DataOnCallEnv",
    description=(
        "RL environment for data pipeline debugging. "
        "The agent acts as an on-call analyst debugging broken reports."
    ),
    version="1.0.0",
    docs_url="/docs",   # Swagger UI at /docs — useful for manual testing
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One env instance per worker process
env = DataOnCallEnv()

# ── Request bodies ────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: int = 1

class StepRequest(BaseModel):
    tool: str
    query: str
    reasoning: Optional[str] = None

class StepResponse(BaseModel):
    observation: Observation
    reward: Optional[Reward] = None
    done: bool
    info: dict

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Validator ping. Returns 200 with env metadata."""
    return {
        "status":  "ok",
        "env":     "DataOnCallEnv",
        "version": "1.0.0",
        "tasks":   [1, 2, 3],
        "spec":    "openenv-0.1",
    }

@app.get("/")
def root():
    """Root redirect info."""
    return {
        "name":      "DataOnCallEnv",
        "docs":      "/docs",
        "health":    "/health",
        "endpoints": ["/reset", "/step", "/state", "/tasks"],
    }

@app.get("/tasks")
def list_tasks():
    """List all available tasks with metadata."""
    return {
        "tasks": [
            {
                "id":          t["id"],
                "difficulty":  t["difficulty"],
                "title":       t["title"],
                "optimal_steps": t["optimal_steps"],
            }
            for t in TASKS.values()
        ]
    }

@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest):
    """
    Start a fresh episode for the given task_id (1, 2, or 3).
    Returns the initial observation containing the scenario description.
    """
    try:
        obs = env.reset(task_id=req.task_id)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    """
    Send one agent action. Returns observation + reward (if done).
    Call POST /reset first.
    """
    if env.task_id is None:
        raise HTTPException(
            status_code=400,
            detail="Not initialized. Call POST /reset first."
        )
    if env.done:
        raise HTTPException(
            status_code=400,
            detail="Episode over. Call POST /reset to start a new episode."
        )

    action = Action(
        tool=req.tool,
        query=req.query,
        reasoning=req.reasoning,
    )

    try:
        obs, reward, done, info = env.step(action)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return StepResponse(observation=obs, reward=reward, done=done, info=info)

@app.get("/state", response_model=EnvState)
def state():
    """Return full current episode state."""
    return env.state()