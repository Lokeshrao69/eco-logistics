"""
Eco-Logistics v2 — Pydantic models.

Extends the v1 schemas with:
- Multi-agent role coordination (3 specialized roles + negotiation protocol)
- Receding-horizon replanning (PlanStep + REVISE flag)
- Curriculum learning (CurriculumState tracking task auto-advancement)

All v1 models (Action, Observation, Reward, PendingShipment, WeatherEvent,
TaskDefinition, GraderResult) are preserved unchanged so the v8 trained
LoRA and the existing FastAPI endpoints continue to work end-to-end.

The v2 features are *opt-in*: the new fields default to None / [] / False
so an old v1 client sending an old-style Action still gets the v1 behavior.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

CITIES = ["Seattle", "Chicago", "NYC"]
CITY_PAIRS = {
    ("Seattle", "Chicago"): {"rail_cost": 3.0, "air_cost": 8.0,  "rail_steps": 3, "air_steps": 1, "rail_carbon": 2.0, "air_carbon": 8.0},
    ("Chicago", "NYC"):     {"rail_cost": 2.0, "air_cost": 6.0,  "rail_steps": 2, "air_steps": 1, "rail_carbon": 1.5, "air_carbon": 6.0},
    ("Seattle", "NYC"):     {"rail_cost": 5.0, "air_cost": 12.0, "rail_steps": 3, "air_steps": 1, "rail_carbon": 3.5, "air_carbon": 12.0},
    ("Chicago", "Seattle"): {"rail_cost": 3.0, "air_cost": 8.0,  "rail_steps": 3, "air_steps": 1, "rail_carbon": 2.0, "air_carbon": 8.0},
    ("NYC", "Chicago"):     {"rail_cost": 2.0, "air_cost": 6.0,  "rail_steps": 2, "air_steps": 1, "rail_carbon": 1.5, "air_carbon": 6.0},
    ("NYC", "Seattle"):     {"rail_cost": 5.0, "air_cost": 12.0, "rail_steps": 3, "air_steps": 1, "rail_carbon": 3.5, "air_carbon": 12.0},
}

SELL_PRICE          = 10.0
STORAGE_FEE_PER_UNIT = 0.5
DECAY_RATE          = 0.02
CARBON_TAX_PER_UNIT = 1.5
HEALTHY_STOCK_BONUS = 0.1
RESTOCK_AMOUNT      = 20

# v2-only constants
PLAN_HORIZON_DEFAULT       = 10            # actions per plan emission
REPLAN_INTERVAL_STEPS      = 4             # agent replans every N steps
CURRICULUM_SUCCESS_THRESH  = 0.80          # 80% success → advance level
CURRICULUM_WINDOW_EPISODES = 5             # over last 5 episodes
PLAN_CONSISTENCY_BONUS     = 5.0
NEGOTIATION_SUCCESS_BONUS  = 8.0
TEAM_CARBON_BONUS_MAX      = 15.0
INVALID_NEGOTIATION_PENALTY = -50.0
FORMAT_PENALTY              = -1000.0      # carried over from v1 (anti-hack)


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class SpeedMode(str, Enum):
    AIR  = "Air"
    RAIL = "Rail"


class TaskDifficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


class AgentRole(str, Enum):
    """v2 multi-agent roles. v1 agents use SOLO and are scored as before."""
    SOLO            = "solo"            # backward-compat: single-agent v1 mode
    SEATTLE_MGR     = "seattle_mgr"     # specializes in Seattle inventory & outbound
    CHICAGO_ROUTER  = "chicago_router"  # specializes in Chicago hub routing
    NYC_CARBON      = "nyc_carbon"      # specializes in NYC + global carbon optimization


class NegotiationStatus(str, Enum):
    """Status of a negotiation proposal between two roles."""
    NONE     = "none"      # no proposal made
    PROPOSED = "proposed"  # one role proposed, other hasn't responded yet
    ACCEPTED = "accepted"  # both roles agreed → bonus applies
    REJECTED = "rejected"  # other role declined → no bonus, no penalty


# ─────────────────────────────────────────────────────────────────────────────
# v1 SCHEMAS (UNCHANGED — preserved for backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class PendingShipment(BaseModel):
    origin: str
    destination: str
    amount: float = Field(ge=0)
    steps_remaining: int = Field(ge=0)
    speed_mode: SpeedMode


class WeatherEvent(BaseModel):
    affected_route: tuple[str, str] = ("Chicago", "NYC")
    cost_multiplier: float = 5.0
    steps_remaining: int = 0


class Observation(BaseModel):
    """v1 single-agent observation. v2 clients use MultiAgentObservation."""
    current_inventory: Dict[str, float] = Field(
        description="Units of stock at each warehouse: Seattle, Chicago, NYC."
    )
    pending_shipments: List[PendingShipment] = Field(
        default_factory=list,
        description="In-transit shipments not yet arrived."
    )
    current_demand: Dict[str, float] = Field(
        description="Customer demand at each city this step."
    )
    carbon_credit_balance: float = Field(
        description="Remaining carbon budget. Negative means over-limit."
    )
    step_number: int = Field(ge=0, description="Current simulation step.")
    total_steps: int = Field(description="Total steps in this episode.")
    weather_alert: Optional[str] = Field(
        default=None,
        description="Warning about weather disruptions."
    )
    cumulative_profit: float = Field(default=0.0)
    cumulative_carbon: float = Field(default=0.0)

    @field_validator("current_inventory", "current_demand")
    @classmethod
    def must_have_all_cities(cls, v: Dict[str, float]) -> Dict[str, float]:
        for city in CITIES:
            if city not in v:
                raise ValueError(f"Missing city: {city}")
        return v


class Action(BaseModel):
    """v1 single-agent action.

    v2 EXTENSION (backward-compatible): two optional fields are added at the
    end. Old v1 clients sending {ship_amount, origin_city, destination_city,
    speed_mode} get None defaults and the env treats them as v1 SOLO actions.
    """
    ship_amount: float = Field(ge=0, description="Units to ship (0 = no-op).")
    origin_city: str = Field(description="Source warehouse city.")
    destination_city: str = Field(description="Target warehouse city.")
    speed_mode: SpeedMode = Field(description="Air or Rail.")

    # ─── v2 EXTENSIONS — optional, default None for v1 compatibility ────────
    role: AgentRole = Field(
        default=AgentRole.SOLO,
        description="v2: agent role tag. SOLO = v1 single-agent behavior."
    )
    negotiation_proposal: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "v2: optional proposal to another role. "
            "Schema: {'proposal_to': '<role>', 'offer': {'route': '<origin→dest>', "
            "'mode': 'Rail', 'shared_amount': <float>}}. "
            "If both roles propose compatible offers, env applies negotiation_success_bonus."
        ),
    )

    @field_validator("origin_city", "destination_city")
    @classmethod
    def valid_city(cls, v: str) -> str:
        if v not in CITIES:
            raise ValueError(f"Invalid city '{v}'. Must be one of {CITIES}")
        return v


class Reward(BaseModel):
    """v1 reward breakdown.

    v2 EXTENSION: three optional bonus fields default to 0.0 so old graders
    that compute totals still work without modification.
    """
    sales_revenue:       float = Field(description="Revenue from fulfilled demand.")
    shipping_cost:       float = Field(description="Cost of shipments initiated this step.")
    carbon_penalty:      float = Field(description="Carbon tax penalty.")
    storage_fee:         float = Field(description="Fee for holding excess inventory.")
    healthy_stock_bonus: float = Field(description="+0.1 if all cities ≥ 20 units.")
    total:               float = Field(description="Net reward this step.")

    # ─── v2 EXTENSIONS — default 0.0 ────────────────────────────────────────
    plan_consistency_bonus: float = Field(
        default=0.0,
        description="v2: +5 if revised plan still fulfills original intent."
    )
    negotiation_success: float = Field(
        default=0.0,
        description="v2: +8 if a multi-role coalition formed (bulk rail vs unilateral air)."
    )
    team_carbon_bonus: float = Field(
        default=0.0,
        description="v2: graded global eco-efficiency bonus when total carbon stays under cap."
    )
    invalid_negotiation_penalty: float = Field(
        default=0.0,
        description="v2: -50 if a negotiation_proposal is malformed."
    )


# ─────────────────────────────────────────────────────────────────────────────
# v2-ONLY SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class PlanStep(BaseModel):
    """One step inside a 10-action plan. The agent emits a list of these.

    Identical fields to Action but without role/negotiation — those live
    on the wrapping AgentPlan, not on individual steps.
    """
    ship_amount: float = Field(ge=0)
    origin_city: str
    destination_city: str
    speed_mode: SpeedMode

    @field_validator("origin_city", "destination_city")
    @classmethod
    def valid_city(cls, v: str) -> str:
        if v not in CITIES:
            raise ValueError(f"Invalid city '{v}'. Must be one of {CITIES}")
        return v


class AgentPlan(BaseModel):
    """A full plan submission from one agent (one role).

    Used for upfront planning AND for replanning. When `is_revision=True`,
    `steps` should cover only the *remaining* steps in the episode.
    """
    role: AgentRole = Field(default=AgentRole.SOLO)
    steps: List[PlanStep] = Field(min_length=1)
    is_revision: bool = Field(
        default=False,
        description="True if this plan is a mid-episode revision under a REVISE flag."
    )
    starting_step: int = Field(
        default=0,
        ge=0,
        description="Step index this plan starts at (0 for initial, k>0 for revisions)."
    )
    negotiation_proposal: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Same schema as Action.negotiation_proposal — can be set on the plan as a whole."
    )


class MultiAgentObservation(BaseModel):
    """v2 observation passed to a multi-agent rollout.

    Carries the v1 Observation unchanged plus the additional context each
    role needs: their own role, what other roles recently did, replanning
    state, and curriculum level.
    """
    base: Observation = Field(description="The underlying v1 observation.")
    role: AgentRole = Field(description="Which role this observation is being sent to.")

    other_agents_recent_actions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Last action emitted by each other role (for coordination). "
            "Each entry: {'role': '<role>', 'step': <int>, 'action': <Action.dict()>}"
        ),
    )

    # ─── Replanning state ────────────────────────────────────────────────────
    revise_plan_flag: bool = Field(
        default=False,
        description="True if the env is asking the agent to emit a revised plan now."
    )
    original_plan_remaining: List[PlanStep] = Field(
        default_factory=list,
        description="The actions from the agent's original plan that haven't run yet."
    )

    # ─── Curriculum state ────────────────────────────────────────────────────
    curriculum_level: int = Field(
        default=0,
        ge=0,
        le=2,
        description="0=restock_only, 1=inventory_balanced, 2=net_zero_profit",
    )
    curriculum_recent_success_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of last N episodes that passed at the current level.",
    )

    # ─── Negotiation state ───────────────────────────────────────────────────
    open_proposals: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Proposals from other roles awaiting this agent's response.",
    )
    last_negotiation_status: NegotiationStatus = Field(
        default=NegotiationStatus.NONE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CURRICULUM TRACKING
# ─────────────────────────────────────────────────────────────────────────────

CURRICULUM_LEVELS: List[str] = ["restock_only", "inventory_balanced", "net_zero_profit"]


class CurriculumState(BaseModel):
    """Persistent curriculum-tracking object held server-side.

    Auto-advances to the next level when the agent's recent_window success
    rate crosses CURRICULUM_SUCCESS_THRESH.
    """
    current_level: int = Field(default=0, ge=0, le=2)
    episodes_at_level: int = Field(default=0, ge=0)
    recent_outcomes: List[bool] = Field(
        default_factory=list,
        description="True/False per episode at current level (capped at CURRICULUM_WINDOW_EPISODES)."
    )

    def record_episode(self, success: bool) -> bool:
        """Record an episode outcome. Returns True if level advanced."""
        self.episodes_at_level += 1
        self.recent_outcomes.append(success)
        if len(self.recent_outcomes) > CURRICULUM_WINDOW_EPISODES:
            self.recent_outcomes = self.recent_outcomes[-CURRICULUM_WINDOW_EPISODES:]

        if (
            self.current_level < 2
            and len(self.recent_outcomes) >= CURRICULUM_WINDOW_EPISODES
            and self.success_rate() >= CURRICULUM_SUCCESS_THRESH
        ):
            self.current_level += 1
            self.episodes_at_level = 0
            self.recent_outcomes = []
            return True
        return False

    def success_rate(self) -> float:
        if not self.recent_outcomes:
            return 0.0
        return sum(self.recent_outcomes) / len(self.recent_outcomes)

    def current_task_id(self) -> str:
        return CURRICULUM_LEVELS[self.current_level]


# ─────────────────────────────────────────────────────────────────────────────
# TASK DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

class TaskDefinition(BaseModel):
    id: str
    name: str
    description: str
    difficulty: TaskDifficulty
    total_steps: int
    carbon_budget: float
    initial_inventory: Dict[str, float]
    demand_profile: str
    # v2: explicit success threshold for curriculum auto-advance
    success_threshold: float = Field(
        default=0.5,
        description="Grader score >= this counts as a 'success' for curriculum tracking."
    )


# v2 task definitions extend the v1 horizons to 25 steps as requested.
# v1 TASKS dict (10/15/20 steps) is preserved for backward compat below.
TASKS_V2: Dict[str, TaskDefinition] = {
    "restock_only": TaskDefinition(
        id="restock_only",
        name="Restock Only (v2, 25-step)",
        description="Maintain stock above 20 units at every warehouse for 25 steps. Stable demand. Curriculum entry point.",
        difficulty=TaskDifficulty.EASY,
        total_steps=25,
        carbon_budget=400.0,
        initial_inventory={"Seattle": 50.0, "Chicago": 50.0, "NYC": 50.0},
        demand_profile="stable",
        success_threshold=0.6,
    ),
    "inventory_balanced": TaskDefinition(
        id="inventory_balanced",
        name="Inventory Balanced (v2, 25-step)",
        description="Keep stock levels within 10% across cities for 25 steps under seasonal demand. Curriculum middle.",
        difficulty=TaskDifficulty.MEDIUM,
        total_steps=25,
        carbon_budget=350.0,
        initial_inventory={"Seattle": 60.0, "Chicago": 40.0, "NYC": 80.0},
        demand_profile="seasonal",
        success_threshold=0.5,
    ),
    "net_zero_profit": TaskDefinition(
        id="net_zero_profit",
        name="Net-Zero Profit (v2, 25-step)",
        description="Maximize profit with carbon ≤ 100 over 25 volatile steps with weather and disruptions. Curriculum hardest.",
        difficulty=TaskDifficulty.HARD,
        total_steps=25,
        carbon_budget=100.0,
        initial_inventory={"Seattle": 40.0, "Chicago": 40.0, "NYC": 40.0},
        demand_profile="volatile",
        success_threshold=0.4,
    ),
}

# v1 TASKS — UNCHANGED. Required by the existing v8 LoRA + main.py.
TASKS = {
    "restock_only": TaskDefinition(
        id="restock_only",
        name="Restock Only",
        description="Maintain stock above 20 units at every warehouse for 10 steps. Demand is low and stable.",
        difficulty=TaskDifficulty.EASY,
        total_steps=10,
        carbon_budget=200.0,
        initial_inventory={"Seattle": 50.0, "Chicago": 50.0, "NYC": 50.0},
        demand_profile="stable",
    ),
    "inventory_balanced": TaskDefinition(
        id="inventory_balanced",
        name="Inventory Balanced",
        description="Keep stock levels within 10% of each other across all three cities for 15 steps under seasonal demand.",
        difficulty=TaskDifficulty.MEDIUM,
        total_steps=15,
        carbon_budget=300.0,
        initial_inventory={"Seattle": 60.0, "Chicago": 40.0, "NYC": 80.0},
        demand_profile="seasonal",
    ),
    "net_zero_profit": TaskDefinition(
        id="net_zero_profit",
        name="Net-Zero Profit",
        description="Maximize profit while keeping total carbon emissions below 80 units over 20 steps. Demand is volatile with weather disruptions.",
        difficulty=TaskDifficulty.HARD,
        total_steps=20,
        carbon_budget=80.0,
        initial_inventory={"Seattle": 40.0, "Chicago": 40.0, "NYC": 40.0},
        demand_profile="volatile",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# GRADER RESULT (v1 unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class GraderResult(BaseModel):
    task_id: str
    score: float = Field(gt=0.0, lt=1.0)
    feedback: str
    metrics: Dict[str, float] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# v2 STEP REQUEST/RESPONSE (used only by /step_v2 endpoint)
# ─────────────────────────────────────────────────────────────────────────────

class V2StepRequest(BaseModel):
    """A v2 step can carry actions from one role (single-agent) or multiple
    roles (multi-agent coordination).

    actions: a list — typically length 1 for single-agent, length 2-3 for
    multi-agent. The env resolves them in order, then computes shared bonuses.
    """
    actions: List[Action] = Field(min_length=1, max_length=3)


class V2StepResponse(BaseModel):
    """v2 step response carries per-role observations + the shared reward."""
    observations: Dict[str, MultiAgentObservation] = Field(
        description="One MultiAgentObservation per role that participated."
    )
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# BACKWARD-COMPAT GUARANTEES
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. All v1 schemas (Action, Observation, Reward, PendingShipment,
#    WeatherEvent, TaskDefinition, GraderResult) accept exactly the same
#    JSON inputs and produce exactly the same JSON outputs as v1, because
#    every new field has a safe default.
#
# 2. The v1 TASKS dict is preserved verbatim. v2 tasks are in TASKS_V2.
#
# 3. The v8 LoRA emitting a v1-style Action JSON will continue to work via
#    the existing /reset and /step endpoints. v2 endpoints are added in
#    main.py at /reset_v2 and /step_v2 (Step 2 of the build).
#
# 4. The Reward total field still represents v1 net reward by default. v2
#    bonuses can be added on top of it explicitly when computing v2 totals,
#    so v1 graders that read .total are unaffected.