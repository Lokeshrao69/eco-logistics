cd ~/Documents/Hackathon
cat > main.py << 'ENDOFFILE'
"""
Eco-Logistics — FastAPI OpenEnv Server

Endpoints: /step, /reset, /state, /tasks, /grader, /baseline
Runs on port 7860 for Hugging Face Spaces.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import EcoLogisticsEnv
from models import (
    TASKS,
    Action,
    GraderResult,
    Observation,
    Reward,
    SpeedMode,
)

app = FastAPI(
    title="Eco-Logistics: Multi-City Supply Chain Optimizer",
    version="1.0.0",
    description="OpenEnv-compliant RL environment for supply chain optimization with carbon tracking.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = EcoLogisticsEnv()


class ResetRequest(BaseModel):
    task_id: str = "restock_only"
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    ship_amount: float = 0.0
    origin_city: str = "Seattle"
    destination_city: str = "Chicago"
    speed_mode: str = "Rail"


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = {}


class GradeRequest(BaseModel):
    task_id: Optional[str] = None


class BaselineRequest(BaseModel):
    task_id: str = "restock_only"
    seed: Optional[int] = 42


class BaselineStepLog(BaseModel):
    step: int
    action: Dict[str, Any]
    reward_total: float
    inventory: Dict[str, float]


class BaselineResponse(BaseModel):
    task_id: str
    grade: GraderResult
    steps: list[BaselineStepLog]


@app.get("/")
def root():
    return {
        "name": "Eco-Logistics: Multi-City Supply Chain Optimizer",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline"],
    }


@app.post("/reset", response_model=Observation)
def reset(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest()
    try:
        obs = env.reset(task_id=req.task_id, seed=req.seed)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(req: Optional[StepRequest] = None):
    if req is None:
        req = StepRequest()
    try:
        mode = SpeedMode(req.speed_mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"speed_mode must be 'Air' or 'Rail', got '{req.speed_mode}'")

    action = Action(
        ship_amount=req.ship_amount,
        origin_city=req.origin_city,
        destination_city=req.destination_city,
        speed_mode=mode,
    )
    try:
        obs, reward, done, info = env.step(action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    return env.state()


@app.get("/tasks")
def tasks():
    return {tid: t.model_dump() for tid, t in TASKS.items()}


@app.post("/grader", response_model=GraderResult)
def grader(req: Optional[GradeRequest] = None):
    if req is None:
        req = GradeRequest()
    try:
        return env.grade(req.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/baseline", response_model=BaselineResponse)
def baseline(req: BaselineRequest):
    from baseline import run_heuristic_baseline
    result = run_heuristic_baseline(task_id=req.task_id, seed=req.seed)
    return result


@app.get("/health")
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
ENDOFFILE