"""
Eco-Logistics — FastAPI OpenEnv Server (v2 — parallel-rollout safe)

Key change from v1: instead of ONE global env, we maintain a session-id-keyed
pool. Each parallel rollout passes its own session_id header and gets its own
env instance. This is required for GRPO training where 16+ rollouts run at once.

Sessions fall back to a default "main" session when no header is provided, so
the existing single-session clients (Swagger, inference.py) keep working.

Endpoints: /step, /reset, /state, /tasks, /grader, /baseline, /health
Runs on port 7860 for Hugging Face Spaces.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException
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
    version="2.0.0",
    description="OpenEnv-compliant RL environment with session-based parallel rollout support.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Session-based env pool ───────────────────────────────────────────────────

MAX_SESSIONS = 32                 # hard cap — blocks rollouts past this
SESSION_TTL_SECONDS = 600         # evict idle envs after 10 min

_sessions: Dict[str, EcoLogisticsEnv] = {}
_last_touch: Dict[str, float] = {}
_pool_lock = threading.Lock()


def _gc_sessions() -> None:
    """Evict idle sessions. Called opportunistically on every request."""
    now = time.time()
    stale = [sid for sid, t in _last_touch.items() if now - t > SESSION_TTL_SECONDS]
    for sid in stale:
        _sessions.pop(sid, None)
        _last_touch.pop(sid, None)


def get_env(session_id: str) -> EcoLogisticsEnv:
    """Return the env for this session, creating it on first use.

    Thread-safe: protected by _pool_lock. FastAPI's default sync handlers run
    in a threadpool, so we need this guard.
    """
    with _pool_lock:
        _gc_sessions()
        if session_id not in _sessions:
            if len(_sessions) >= MAX_SESSIONS:
                raise HTTPException(
                    status_code=429,
                    detail=f"Too many concurrent sessions (max {MAX_SESSIONS}). "
                           f"Increase MAX_SESSIONS in main.py or wait for idle sessions to evict.",
                )
            _sessions[session_id] = EcoLogisticsEnv()
        _last_touch[session_id] = time.time()
        return _sessions[session_id]


# ── Request / Response Models ────────────────────────────────────────────────

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


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Eco-Logistics: Multi-City Supply Chain Optimizer",
        "version": "2.0.0",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline", "/health", "/sessions"],
        "parallel_rollouts": f"supported (max {MAX_SESSIONS} concurrent sessions)",
    }


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest, x_session_id: str = Header(default="main")):
    """Reset the environment. Pass X-Session-Id header for parallel rollouts."""
    try:
        env = get_env(x_session_id)
        obs = env.reset(task_id=req.task_id, seed=req.seed)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest, x_session_id: str = Header(default="main")):
    """Execute one step. Pass X-Session-Id header for parallel rollouts."""
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
        env = get_env(x_session_id)
        obs, reward, done = env.step(action)
        return StepResponse(observation=obs, reward=reward, done=done)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state(x_session_id: str = Header(default="main")):
    """Return full environment state snapshot for this session."""
    env = get_env(x_session_id)
    return env.state()


@app.get("/tasks")
def tasks():
    """List all available tasks."""
    return {tid: t.model_dump() for tid, t in TASKS.items()}


@app.post("/grader", response_model=GraderResult)
def grader(req: GradeRequest, x_session_id: str = Header(default="main")):
    """Grade the current episode for this session."""
    try:
        env = get_env(x_session_id)
        return env.grade(req.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/baseline", response_model=BaselineResponse)
def baseline(req: BaselineRequest):
    """Run a deterministic heuristic baseline and return results.
    Note: baseline uses its own ephemeral env, not the session pool.
    """
    from baseline import run_heuristic_baseline
    result = run_heuristic_baseline(task_id=req.task_id, seed=req.seed)
    return result


@app.get("/sessions")
def sessions():
    """Diagnostic endpoint — shows active session count and IDs."""
    with _pool_lock:
        _gc_sessions()
        return {
            "active": len(_sessions),
            "max": MAX_SESSIONS,
            "session_ids": list(_sessions.keys()),
            "idle_ttl_seconds": SESSION_TTL_SECONDS,
        }


@app.get("/health")
def health():
    return {"status": "healthy", "active_sessions": len(_sessions)}


# ── Startup ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)