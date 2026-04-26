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
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import EcoLogisticsEnv
from models import (
    TASKS,
    TASKS_V2,
    Action,
    AgentPlan,
    AgentRole,
    CurriculumState,
    GraderResult,
    MultiAgentObservation,
    NegotiationStatus,
    Observation,
    PlanStep,
    Reward,
    SpeedMode,
    V2StepRequest,
    V2StepResponse,
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

MAX_SESSIONS = 64                # hard cap — blocks rollouts past this
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
        obs, reward, done, info = env.step(action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
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
    
@app.post("/reset_all")
def reset_all():
    """Admin endpoint — wipe the entire session pool.
 
    Useful when a training run gets stuck or leaves stale sessions around.
    Much faster than a full Space restart.
    """
    with _pool_lock:
        count = len(_sessions)
        _sessions.clear()
        _last_touch.clear()
        return {
            "cleared": count,
            "active_after": len(_sessions),
            "message": f"Pool cleared. {count} sessions evicted.",
        }


# ── v2 ENDPOINTS (curriculum + replanning + multi-agent) ────────────────────
# These add new functionality WITHOUT touching v1 endpoints. The v8 LoRA's
# rollout loop (which calls /reset and /step) keeps working unchanged.

class V2ResetRequest(BaseModel):
    task_id: str = "restock_only"
    seed: Optional[int] = 42
    use_v2: bool = True   # default to True for v2-flavored endpoints


class V2ResetCurriculumRequest(BaseModel):
    seed: Optional[int] = 42


class V2SubmitPlanRequest(BaseModel):
    plan: AgentPlan


class V2SubmitMultiagentPlansRequest(BaseModel):
    plans: Dict[str, AgentPlan]


class V2RunChunkResponse(BaseModel):
    observation: Optional[Observation] = None
    rewards: List[Reward] = []
    done: bool = False
    info: Dict[str, Any] = {}


class V2RunMultiagentChunkResponse(BaseModel):
    observations: Dict[str, MultiAgentObservation] = {}
    rewards: List[Reward] = []
    done: bool = False
    info: Dict[str, Any] = {}


@app.post("/reset_v2", response_model=Observation)
def reset_v2(req: V2ResetRequest, x_session_id: str = Header(default="main")):
    """v2 reset. Set use_v2=True for 25-step variants from TASKS_V2."""
    try:
        env = get_env(x_session_id)
        return env.reset(task_id=req.task_id, seed=req.seed, use_v2=req.use_v2)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/reset_curriculum", response_model=Observation)
def reset_curriculum(req: V2ResetCurriculumRequest, x_session_id: str = Header(default="main")):
    """Reset to whatever level the curriculum is currently at (uses TASKS_V2)."""
    env = get_env(x_session_id)
    return env.reset_to_curriculum_level(seed=req.seed)


@app.get("/curriculum_state")
def curriculum_state(x_session_id: str = Header(default="main")):
    """Inspect the current curriculum state for this session."""
    env = get_env(x_session_id)
    cs = env.get_curriculum_state()
    return {
        "current_level": cs.current_level,
        "current_task_id": cs.current_task_id(),
        "episodes_at_level": cs.episodes_at_level,
        "success_rate": round(cs.success_rate(), 3),
        "recent_outcomes": cs.recent_outcomes,
    }


@app.post("/force_curriculum_level")
def force_curriculum_level(level: int, x_session_id: str = Header(default="main")):
    """Manually set the curriculum level (0/1/2). For evaluation runs."""
    env = get_env(x_session_id)
    try:
        env.force_curriculum_level(level)
        return {"ok": True, "level": level}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/submit_plan")
def submit_plan(req: V2SubmitPlanRequest, x_session_id: str = Header(default="main")):
    """Agent submits an upfront 10/25-step plan."""
    env = get_env(x_session_id)
    try:
        env.submit_plan(req.plan)
        return {"ok": True, "plan_length": len(req.plan.steps)}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/submit_revised_plan")
def submit_revised_plan(req: V2SubmitPlanRequest, x_session_id: str = Header(default="main")):
    """Agent submits a revised plan (mid-episode replan)."""
    env = get_env(x_session_id)
    try:
        env.submit_revised_plan(req.plan)
        return {"ok": True, "revisions_so_far": env._plan_revisions_count}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/run_chunk", response_model=V2RunChunkResponse)
def run_chunk(
    chunk_size: Optional[int] = None,
    x_session_id: str = Header(default="main"),
):
    """Run the next chunk of plan steps. Default = REPLAN_INTERVAL_STEPS = 4."""
    env = get_env(x_session_id)
    try:
        obs, rewards, done, info = env.run_plan_chunk(max_chunk_steps=chunk_size)
        return V2RunChunkResponse(
            observation=obs,
            rewards=rewards,
            done=done,
            info=info,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/replanning_state")
def replanning_state(x_session_id: str = Header(default="main")):
    """Inspect the current single-agent replanning state."""
    env = get_env(x_session_id)
    return env.get_replanning_state()


@app.post("/submit_multiagent_plans")
def submit_multiagent_plans(
    req: V2SubmitMultiagentPlansRequest,
    x_session_id: str = Header(default="main"),
):
    """Submit one plan per role. Triggers immediate negotiation resolution."""
    env = get_env(x_session_id)
    try:
        env.submit_multiagent_plans(req.plans)
        return {
            "ok": True,
            "active_roles": list(req.plans.keys()),
            "negotiation_status": {
                r: s.value for r, s in env._role_negotiation_status.items()
            },
            "cumulative_negotiation_bonus": round(env._cumulative_negotiation_bonus, 2),
        }
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/run_multiagent_chunk", response_model=V2RunMultiagentChunkResponse)
def run_multiagent_chunk(
    chunk_size: Optional[int] = None,
    x_session_id: str = Header(default="main"),
):
    """Run one chunk of multi-agent steps."""
    env = get_env(x_session_id)
    try:
        obs_per_role, rewards, done, info = env.run_multiagent_chunk(max_chunk_steps=chunk_size)
        return V2RunMultiagentChunkResponse(
            observations=obs_per_role,
            rewards=rewards,
            done=done,
            info=info,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/multiagent_state")
def multiagent_state(x_session_id: str = Header(default="main")):
    """Inspect multi-agent state for this session."""
    env = get_env(x_session_id)
    return env.get_multiagent_state()


@app.get("/tasks_v2")
def tasks_v2():
    """List the v2 (25-step) task variants."""
    return {tid: t.model_dump() for tid, t in TASKS_V2.items()}


@app.get("/health")
def health():
    return {"status": "healthy", "active_sessions": len(_sessions)}


# ── Startup ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)