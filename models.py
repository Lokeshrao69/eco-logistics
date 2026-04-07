"""
Eco-Logistics: Multi-City Supply Chain Optimizer
Pydantic models for the OpenEnv environment.
"""
 
from __future__ import annotations
 
from enum import Enum
from typing import Dict, List, Optional
 
from pydantic import BaseModel, Field, field_validator
 
 
# ── Constants ────────────────────────────────────────────────────────────────
 
CITIES = ["Seattle", "Chicago", "NYC"]
CITY_PAIRS = {
    ("Seattle", "Chicago"): {"rail_cost": 3.0, "air_cost": 8.0, "rail_steps": 3, "air_steps": 1, "rail_carbon": 2.0, "air_carbon": 8.0},
    ("Chicago", "NYC"):     {"rail_cost": 2.0, "air_cost": 6.0, "rail_steps": 2, "air_steps": 1, "rail_carbon": 1.5, "air_carbon": 6.0},
    ("Seattle", "NYC"):     {"rail_cost": 5.0, "air_cost": 12.0, "rail_steps": 3, "air_steps": 1, "rail_carbon": 3.5, "air_carbon": 12.0},
    ("Chicago", "Seattle"): {"rail_cost": 3.0, "air_cost": 8.0, "rail_steps": 3, "air_steps": 1, "rail_carbon": 2.0, "air_carbon": 8.0},
    ("NYC", "Chicago"):     {"rail_cost": 2.0, "air_cost": 6.0, "rail_steps": 2, "air_steps": 1, "rail_carbon": 1.5, "air_carbon": 6.0},
    ("NYC", "Seattle"):     {"rail_cost": 5.0, "air_cost": 12.0, "rail_steps": 3, "air_steps": 1, "rail_carbon": 3.5, "air_carbon": 12.0},
}
 
SELL_PRICE = 10.0
STORAGE_FEE_PER_UNIT = 0.5
DECAY_RATE = 0.02  # 2% per step
CARBON_TAX_PER_UNIT = 1.5
HEALTHY_STOCK_BONUS = 0.1
RESTOCK_AMOUNT = 20  # units produced per city per step
 
 
# ── Enums ────────────────────────────────────────────────────────────────────
 
class SpeedMode(str, Enum):
    AIR = "Air"
    RAIL = "Rail"
 
 
class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
 
 
# ── Shipment Tracking ────────────────────────────────────────────────────────
 
class PendingShipment(BaseModel):
    origin: str
    destination: str
    amount: float = Field(ge=0)
    steps_remaining: int = Field(ge=0)
    speed_mode: SpeedMode
 
 
# ── Weather Event ────────────────────────────────────────────────────────────
 
class WeatherEvent(BaseModel):
    affected_route: tuple[str, str] = ("Chicago", "NYC")
    cost_multiplier: float = 5.0
    steps_remaining: int = 0
 
 
# ── Core OpenEnv Types ───────────────────────────────────────────────────────
 
class Observation(BaseModel):
    """What the agent sees each step."""
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
        description="Warning about weather disruptions (e.g. 'Chicago→NYC route 5x cost for 2 steps')."
    )
    cumulative_profit: float = Field(
        default=0.0,
        description="Running total of profit so far."
    )
    cumulative_carbon: float = Field(
        default=0.0,
        description="Running total of carbon emissions so far."
    )
 
    @field_validator("current_inventory", "current_demand")
    @classmethod
    def must_have_all_cities(cls, v: Dict[str, float]) -> Dict[str, float]:
        for city in CITIES:
            if city not in v:
                raise ValueError(f"Missing city: {city}")
        return v
 
 
class Action(BaseModel):
    """What the agent does each step."""
    ship_amount: float = Field(ge=0, description="Units to ship (0 = no-op).")
    origin_city: str = Field(description="Source warehouse city.")
    destination_city: str = Field(description="Target warehouse city.")
    speed_mode: SpeedMode = Field(description="Shipping mode: Air (fast, expensive, high carbon) or Rail (slow, cheap, low carbon).")
 
    @field_validator("origin_city", "destination_city")
    @classmethod
    def valid_city(cls, v: str) -> str:
        if v not in CITIES:
            raise ValueError(f"Invalid city '{v}'. Must be one of {CITIES}")
        return v
 
 
class Reward(BaseModel):
    """Breakdown of the reward signal for one step."""
    sales_revenue: float = Field(description="Revenue from fulfilled demand.")
    shipping_cost: float = Field(description="Cost of shipments initiated this step.")
    carbon_penalty: float = Field(description="Carbon tax penalty for emissions.")
    storage_fee: float = Field(description="Fee for holding excess inventory.")
    healthy_stock_bonus: float = Field(description="+0.1 if all cities above 20 units.")
    total: float = Field(description="Net reward this step.")
 
 
# ── Task Definitions ─────────────────────────────────────────────────────────
 
class TaskDefinition(BaseModel):
    id: str
    name: str
    description: str
    difficulty: TaskDifficulty
    total_steps: int
    carbon_budget: float = Field(description="Total allowed carbon emissions.")
    initial_inventory: Dict[str, float]
    demand_profile: str = Field(description="How demand fluctuates: 'stable', 'seasonal', 'volatile'.")
 
 
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
 
 
# ── Grader Result ────────────────────────────────────────────────────────────
 
class GraderResult(BaseModel):
    task_id: str
    score: float = Field(gt=0.0, lt=1.0, description="Score strictly between 0.0 and 1.0 (exclusive).")
    feedback: str
    metrics: Dict[str, float] = Field(default_factory=dict)