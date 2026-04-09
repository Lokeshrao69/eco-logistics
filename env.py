"""
Eco-Logistics Environment — Core simulation engine.

Implements step(), reset(), state() and deterministic graders for all 3 tasks.
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from models import (
    CITIES,
    CITY_PAIRS,
    CARBON_TAX_PER_UNIT,
    DECAY_RATE,
    HEALTHY_STOCK_BONUS,
    RESTOCK_AMOUNT,
    SELL_PRICE,
    STORAGE_FEE_PER_UNIT,
    TASKS,
    Action,
    GraderResult,
    Observation,
    PendingShipment,
    Reward,
    SpeedMode,
    TaskDefinition,
    WeatherEvent,
)


class EcoLogisticsEnv:
    """Multi-city supply chain environment with carbon tracking."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._seed = seed
        self._task: Optional[TaskDefinition] = None
        self._inventory: Dict[str, float] = {}
        self._pending: List[PendingShipment] = []
        self._weather: WeatherEvent = WeatherEvent()
        self._step_num: int = 0
        self._cumulative_profit: float = 0.0
        self._cumulative_carbon: float = 0.0
        self._carbon_budget: float = 0.0
        self._history: List[Dict[str, Any]] = []
        self._done: bool = False

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self, task_id: str, seed: Optional[int] = None) -> Observation:
        if task_id not in TASKS:
            raise ValueError(f"Unknown task: {task_id}. Choose from {list(TASKS.keys())}")

        self._rng = random.Random(seed if seed is not None else self._seed)
        self._task = TASKS[task_id]
        self._inventory = deepcopy(self._task.initial_inventory)
        self._pending = []
        self._weather = WeatherEvent()
        self._step_num = 0
        self._cumulative_profit = 0.0
        self._cumulative_carbon = 0.0
        self._carbon_budget = self._task.carbon_budget
        self._history = []
        self._done = False
        self._last_demand: Dict[str, float] = {}
        self._last_fulfilled: Dict[str, float] = {}

        return self._make_observation()

    # ── Step ─────────────────────────────────────────────────────────────

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._task is None:
            raise RuntimeError("Call reset() before step()")
        if self._done:
            raise RuntimeError("Episode is done. Call reset().")

        reward = self._process_step(action)
        self._step_num += 1

        if self._step_num >= self._task.total_steps:
            self._done = True

        obs = self._make_observation()

        # Build info dict (OpenEnv spec: step returns obs, reward, done, info)
        info: Dict[str, Any] = {
            "step_number": self._step_num,
            "cumulative_profit": round(self._cumulative_profit, 2),
            "cumulative_carbon": round(self._cumulative_carbon, 2),
            "carbon_budget_remaining": round(self._carbon_budget - self._cumulative_carbon, 2),
            "weather_active": self._weather.steps_remaining > 0,
            "pending_shipment_count": len(self._pending),
        }

        # Record history for grading
        self._history.append({
            "step": self._step_num,
            "inventory": deepcopy(self._inventory),
            "action": action.model_dump(),
            "reward": reward.model_dump(),
            "cumulative_profit": self._cumulative_profit,
            "cumulative_carbon": self._cumulative_carbon,
            "demand": deepcopy(self._last_demand),
            "fulfilled": deepcopy(self._last_fulfilled),
        })

        return obs, reward, self._done, info

    def _process_step(self, action: Action) -> Reward:
        """Execute one tick of the simulation."""

        # 1. Generate demand
        demand = self._generate_demand()

        # 2. Trigger / tick weather events
        self._tick_weather()

        # 3. Deliver arrived shipments
        self._deliver_shipments()

        # 4. Apply inventory decay
        for city in CITIES:
            self._inventory[city] *= (1.0 - DECAY_RATE)

        # 5. Local production (restock)
        for city in CITIES:
            self._inventory[city] += RESTOCK_AMOUNT

        # 6. Process the agent's shipping action
        shipping_cost, carbon_emitted = self._process_action(action)

        # 7. Fulfill demand → revenue
        sales_revenue = 0.0
        fulfilled = {}
        for city in CITIES:
            sold = min(self._inventory[city], demand[city])
            sales_revenue += sold * SELL_PRICE
            self._inventory[city] -= sold
            fulfilled[city] = sold

        # Store for history tracking
        self._last_demand = demand
        self._last_fulfilled = fulfilled

        # 8. Storage fee
        storage_fee = sum(max(0, v) for v in self._inventory.values()) * STORAGE_FEE_PER_UNIT

        # 9. Carbon penalty
        self._cumulative_carbon += carbon_emitted
        carbon_penalty = carbon_emitted * CARBON_TAX_PER_UNIT

        # 10. Healthy stock bonus
        healthy = all(self._inventory[c] >= 20.0 for c in CITIES)
        bonus = HEALTHY_STOCK_BONUS if healthy else 0.0

        total = sales_revenue - shipping_cost - carbon_penalty - storage_fee + bonus
        self._cumulative_profit += total

        return Reward(
            sales_revenue=round(sales_revenue, 2),
            shipping_cost=round(shipping_cost, 2),
            carbon_penalty=round(carbon_penalty, 2),
            storage_fee=round(storage_fee, 2),
            healthy_stock_bonus=round(bonus, 2),
            total=round(total, 2),
        )

    # ── Demand Generation ────────────────────────────────────────────────

    def _generate_demand(self) -> Dict[str, float]:
        profile = self._task.demand_profile
        demand: Dict[str, float] = {}

        for city in CITIES:
            if profile == "stable":
                base = 10.0 + self._rng.gauss(0, 1)
            elif profile == "seasonal":
                wave = 8.0 * math.sin(2 * math.pi * self._step_num / 7)
                base = 15.0 + wave + self._rng.gauss(0, 2)
            elif profile == "surge":
                # NYC demand spikes 3x after step 7
                base_demand = 12.0 + self._rng.gauss(0, 2)
                if city == "NYC" and self._step_num >= 7:
                    base = base_demand * 3.0
                else:
                    base = base_demand
            else:  # volatile
                base = 12.0 + self._rng.gauss(0, 6)

            demand[city] = max(0.0, round(base, 1))

        return demand

    # ── Weather Events ───────────────────────────────────────────────────

    def _tick_weather(self):
        """Decrement active weather or randomly spawn a new one."""
        if self._weather.steps_remaining > 0:
            self._weather.steps_remaining -= 1
        else:
            # 15% chance each step in volatile mode, 5% otherwise
            chance = 0.15 if (self._task and self._task.demand_profile == "volatile") else 0.05
            if self._rng.random() < chance:
                routes = [("Chicago", "NYC"), ("Seattle", "Chicago"), ("Seattle", "NYC")]
                route = self._rng.choice(routes)
                self._weather = WeatherEvent(
                    affected_route=route,
                    cost_multiplier=5.0,
                    steps_remaining=2,
                )

    # ── Shipment Processing ──────────────────────────────────────────────

    def _deliver_shipments(self):
        arrived = [s for s in self._pending if s.steps_remaining <= 0]
        self._pending = [s for s in self._pending if s.steps_remaining > 0]

        for s in arrived:
            self._inventory[s.destination] += s.amount

        # Tick remaining
        for s in self._pending:
            s.steps_remaining -= 1

    def _process_action(self, action: Action) -> Tuple[float, float]:
        """Validate and queue the agent's shipment. Returns (cost, carbon)."""
        if action.ship_amount <= 0:
            return 0.0, 0.0

        if action.origin_city == action.destination_city:
            return 0.0, 0.0

        route = (action.origin_city, action.destination_city)
        if route not in CITY_PAIRS:
            return 0.0, 0.0

        # Clamp to available inventory
        actual = min(action.ship_amount, self._inventory[action.origin_city])
        if actual <= 0:
            return 0.0, 0.0

        info = CITY_PAIRS[route]
        is_air = action.speed_mode == SpeedMode.AIR

        base_cost = info["air_cost"] if is_air else info["rail_cost"]
        lead_time = info["air_steps"] if is_air else info["rail_steps"]
        carbon = info["air_carbon"] if is_air else info["rail_carbon"]

        # Weather multiplier
        fwd = route
        rev = (route[1], route[0])
        if self._weather.steps_remaining > 0 and (
            fwd == self._weather.affected_route or rev == self._weather.affected_route
        ):
            base_cost *= self._weather.cost_multiplier

        cost = base_cost * actual
        carbon_total = carbon * actual

        # Deduct from origin
        self._inventory[action.origin_city] -= actual

        # Queue shipment
        self._pending.append(PendingShipment(
            origin=action.origin_city,
            destination=action.destination_city,
            amount=actual,
            steps_remaining=lead_time,
            speed_mode=action.speed_mode,
        ))

        return round(cost, 2), round(carbon_total, 2)

    # ── Observation Builder ──────────────────────────────────────────────

    def _make_observation(self) -> Observation:
        demand = self._generate_demand()
        alert = None
        if self._weather.steps_remaining > 0:
            r = self._weather.affected_route
            alert = f"{r[0]}→{r[1]} route cost is {self._weather.cost_multiplier}x for {self._weather.steps_remaining} more step(s)."

        return Observation(
            current_inventory={c: round(self._inventory[c], 1) for c in CITIES},
            pending_shipments=[s.model_copy() for s in self._pending],
            current_demand={c: round(demand[c], 1) for c in CITIES},
            carbon_credit_balance=round(self._carbon_budget - self._cumulative_carbon, 1),
            step_number=self._step_num,
            total_steps=self._task.total_steps,
            weather_alert=alert,
            cumulative_profit=round(self._cumulative_profit, 2),
            cumulative_carbon=round(self._cumulative_carbon, 2),
        )

    # ── State (full snapshot for OpenEnv) ────────────────────────────────

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self._task.id if self._task else None,
            "step_number": self._step_num,
            "done": self._done,
            "inventory": deepcopy(self._inventory),
            "pending_shipments": [s.model_dump() for s in self._pending],
            "cumulative_profit": self._cumulative_profit,
            "cumulative_carbon": self._cumulative_carbon,
            "carbon_budget": self._carbon_budget,
            "weather": {
                "route": list(self._weather.affected_route),
                "multiplier": self._weather.cost_multiplier,
                "steps_remaining": self._weather.steps_remaining,
            },
            "history_length": len(self._history),
        }

    # ── Graders ──────────────────────────────────────────────────────────

    def grade(self, task_id: Optional[str] = None) -> GraderResult:
        tid = task_id or (self._task.id if self._task else None)
        if tid is None:
            raise ValueError("No task to grade.")

        if tid == "restock_only":
            return self._grade_restock()
        elif tid == "inventory_balanced":
            return self._grade_balanced()
        elif tid == "net_zero_profit":
            return self._grade_net_zero()
        elif tid == "demand_surge":
            return self._grade_demand_surge()
        else:
            raise ValueError(f"Unknown task: {tid}")

    @staticmethod
    def _clamp_score(score: float) -> float:
        """Clamp score to strictly between 0.0 and 1.0 (exclusive)."""
        return min(0.999, max(0.001, score))

    def _grade_restock(self) -> GraderResult:
        """Easy: All cities stay above 20 units for every recorded step."""
        if not self._history:
            return GraderResult(task_id="restock_only", score=0.001, feedback="No steps taken.")

        total_checks = 0
        passed_checks = 0

        for h in self._history:
            for city in CITIES:
                total_checks += 1
                if h["inventory"][city] >= 20.0:
                    passed_checks += 1

        score = round(passed_checks / total_checks, 4) if total_checks > 0 else 0.0
        return GraderResult(
            task_id="restock_only",
            score=self._clamp_score(score),
            feedback=f"{passed_checks}/{total_checks} city-step checks passed the 20-unit threshold.",
            metrics={"passed_checks": passed_checks, "total_checks": total_checks},
        )

    def _grade_balanced(self) -> GraderResult:
        """Medium: Inventory levels within 10% of each other across cities.
        
        Scoring uses two components:
        - 60% weight: fraction of steps perfectly balanced (within 10%)
        - 40% weight: average closeness-to-balance across ALL steps (smooth gradient)
        This ensures partial credit for near-balanced states.
        """
        if not self._history:
            return GraderResult(task_id="inventory_balanced", score=0.001, feedback="No steps taken.")

        balanced_steps = 0
        closeness_scores = []

        for h in self._history:
            levels = [h["inventory"][c] for c in CITIES]
            mean_val = sum(levels) / len(levels)
            if mean_val < 1.0:
                closeness_scores.append(0.0)
                continue
            max_dev = max(abs(v - mean_val) / mean_val for v in levels)
            if max_dev <= 0.10:
                balanced_steps += 1
                closeness_scores.append(1.0)
            else:
                # Partial credit: 1.0 at 0% deviation, 0.0 at 50%+ deviation
                closeness = max(0.0, 1.0 - (max_dev / 0.50))
                closeness_scores.append(closeness)

        perfect_ratio = balanced_steps / len(self._history)
        avg_closeness = sum(closeness_scores) / len(closeness_scores) if closeness_scores else 0.0

        score = round(0.6 * perfect_ratio + 0.4 * avg_closeness, 4)
        return GraderResult(
            task_id="inventory_balanced",
            score=self._clamp_score(score),
            feedback=f"{balanced_steps}/{len(self._history)} steps perfectly balanced. Avg closeness: {avg_closeness:.2f}.",
            metrics={
                "balanced_steps": balanced_steps,
                "total_steps": len(self._history),
                "avg_closeness": round(avg_closeness, 4),
            },
        )

    def _grade_net_zero(self) -> GraderResult:
        """Hard: Maximize profit while staying under carbon budget."""
        if not self._history:
            return GraderResult(task_id="net_zero_profit", score=0.001, feedback="No steps taken.")

        carbon_ok = self._cumulative_carbon <= self._carbon_budget
        profit = self._cumulative_profit

        # Normalize profit: baseline expectation ~2000 for 20 steps of pure demand
        # Top-end theoretical ~4000
        max_expected_profit = self._task.total_steps * len(CITIES) * 15.0 * SELL_PRICE
        profit_score = max(0.0, profit) / max_expected_profit if max_expected_profit > 0 else 0.0
        profit_score = min(1.0, profit_score)

        if not carbon_ok:
            # Penalize proportionally to overshoot
            overshoot = (self._cumulative_carbon - self._carbon_budget) / self._carbon_budget
            penalty = min(1.0, overshoot)
            profit_score *= max(0.0, 1.0 - penalty)

        score = round(profit_score, 4)
        return GraderResult(
            task_id="net_zero_profit",
            score=self._clamp_score(score),
            feedback=f"Profit: {profit:.1f} | Carbon: {self._cumulative_carbon:.1f}/{self._carbon_budget} | {'PASS' if carbon_ok else 'OVER BUDGET'}",
            metrics={
                "profit": round(profit, 2),
                "carbon_used": round(self._cumulative_carbon, 2),
                "carbon_budget": self._carbon_budget,
                "carbon_ok": float(carbon_ok),
            },
        )

    def _grade_demand_surge(self) -> GraderResult:
        """Hard: Fulfill ≥85% of total demand during a NYC demand surge, under carbon budget."""
        if not self._history:
            return GraderResult(task_id="demand_surge", score=0.001, feedback="No steps taken.")

        total_demand = 0.0
        total_fulfilled = 0.0

        for h in self._history:
            for city in CITIES:
                d = h.get("demand", {}).get(city, 0.0)
                f = h.get("fulfilled", {}).get(city, 0.0)
                total_demand += d
                total_fulfilled += f

        fill_rate = total_fulfilled / total_demand if total_demand > 0 else 0.0
        carbon_ok = self._cumulative_carbon <= self._carbon_budget

        # Score: fulfillment rate is primary (70%), carbon compliance is secondary (30%)
        fill_score = fill_rate  # 0.0 to 1.0
        carbon_score = 1.0 if carbon_ok else max(0.0, 1.0 - (self._cumulative_carbon - self._carbon_budget) / self._carbon_budget)

        score = round(0.7 * fill_score + 0.3 * carbon_score, 4)
        return GraderResult(
            task_id="demand_surge",
            score=self._clamp_score(score),
            feedback=f"Fill rate: {fill_rate:.1%} (target ≥85%) | Carbon: {self._cumulative_carbon:.1f}/{self._carbon_budget} | {'OK' if carbon_ok else 'OVER'}",
            metrics={
                "total_demand": round(total_demand, 2),
                "total_fulfilled": round(total_fulfilled, 2),
                "fill_rate": round(fill_rate, 4),
                "carbon_used": round(self._cumulative_carbon, 2),
                "carbon_ok": float(carbon_ok),
            },
        )