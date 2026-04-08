"""FastAPI entry point for DataOnCallEnv.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from ..models import Action, Observation, Reward, EnvState
from ..environment import DataOnCallEnv
from ..tasks import TASKS

app = FastAPI(
    title="DataOnCallEnv",
    description=(
        "RL environment for data pipeline debugging. "
        "The agent acts as an on-call analyst debugging broken reports. "
        "Features: partial observability, query costs, realistic logs, "
        "anti-cheat constraints, and tiered evaluation."
    ),
    version="2.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One env instance per worker process
env = DataOnCallEnv()

# Models

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

# Endpoints

@app.get("/health")
def health():
    """Health check endpoint for the environment."""
    return {
        "status":  "ok",
        "env":     "DataOnCallEnv",
        "version": "2.0.0",
        "tasks":   [1, 2, 3],
        "spec":    "openenv-0.1",
        "features": [
            "partial_observability",
            "query_costs",
            "realistic_logs",
            "anti_cheat",
            "tiered_evaluation",
        ],
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
                "id":            t["id"],
                "difficulty":    t["difficulty"],
                "title":         t["title"],
                "optimal_steps": t["optimal_steps"],
                "optimal_cost":  t["optimal_cost"],
            }
            for t in TASKS.values()
        ]
    }

@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest):
    """
    Start a fresh episode for the given task_id (1, 2, or 3).
    Returns the initial observation containing the scenario description.
    Agent must discover tables via list_tables() — not provided in reset.
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
def main():
    import uvicorn
    uvicorn.run("dataoncallenv.api.app:app", host="0.0.0.0", port=7860, reload=True)

if __name__ == "__main__":
    main()
