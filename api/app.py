# api/app.py
# FastAPI server. Wraps the environment in HTTP endpoints.
# The hackathon validator will ping these URLs to confirm your env works.
#
# Endpoints:
#   POST /reset        — start a new episode
#   POST /step         — send an action, get observation back
#   GET  /state        — inspect current episode state
#   GET  /health       — validator ping (must return 200)
#   GET  /tasks        — list available tasks

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys, os

# Make sure Python can find our modules when running from /api
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Action, Observation, Reward, EnvState
from environment import DataOnCallEnv
from tasks import TASKS

app = FastAPI(
    title="DataOnCallEnv",
    description="RL environment for data pipeline debugging agents",
    version="1.0.0",
)

# Allow cross-origin requests — needed for HuggingFace Spaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One environment instance per server process
# In production with multiple workers, each worker has its own instance
env = DataOnCallEnv()

# ── Request/Response models ────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: int = 1  # defaults to task 1 if not specified

class StepRequest(BaseModel):
    tool: str
    query: str
    reasoning: Optional[str] = None

class StepResponse(BaseModel):
    observation: Observation
    reward: Optional[Reward] = None   # None until episode ends
    done: bool
    info: dict

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Validator ping. Must return 200."""
    return {"status": "ok", "env": "DataOnCallEnv", "version": "1.0.0"}

@app.get("/tasks")
def list_tasks():
    """Return metadata about all 3 tasks."""
    return {
        "tasks": [
            {
                "id": t["id"],
                "difficulty": t["difficulty"],
                "title": t["title"],
            }
            for t in TASKS.values()
        ]
    }

@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest):
    """
    Start a fresh episode.
    Rebuilds the database and returns the scenario description.
    """
    try:
        obs = env.reset(task_id=req.task_id)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    """
    Send one agent action. Returns observation and reward (if episode ended).
    """
    if env.task_id is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first."
        )
    if env.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is over. Call POST /reset to start a new episode."
        )

    action = Action(tool=req.tool, query=req.query, reasoning=req.reasoning)
    obs, reward, done, info = env.step(action)

    return StepResponse(observation=obs, reward=reward, done=done, info=info)

@app.get("/state", response_model=EnvState)
def state():
    """Return the full current episode state."""
    return env.state()