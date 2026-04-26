"""
Eco-Logistics v2 — Core simulation engine.

Step 2 deliverable. Adds:
- Curriculum learning: env tracks per-session CurriculumState and exposes
  current level via get_curriculum_state() / advance_or_repeat() helpers.
- 25-step v2 task variants (TASKS_V2) selectable via reset(use_v2=True).

Preserves:
- All v1 reset/step/grade behavior unchanged when called with v1 task ids.
- Existing /reset, /step, /state, /grader endpoints in main.py keep working.
- v8 LoRA against v1 TASKS still works exactly as before.

Steps 3 (replanning) and 4 (multi-agent) extend this same class without
breaking the v1 contract.
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
    INVALID_NEGOTIATION_PENALTY,
    NEGOTIATION_SUCCESS_BONUS,
    PLAN_CONSISTENCY_BONUS,
    REPLAN_INTERVAL_STEPS,
    RESTOCK_AMOUNT,
    SELL_PRICE,
    STORAGE_FEE_PER_UNIT,
    TASKS,
    TASKS_V2,
    TEAM_CARBON_BONUS_MAX,
    Action,
    AgentPlan,
    AgentRole,
    CurriculumState,
    GraderResult,
    MultiAgentObservation,
    NegotiationStatus,
    Observation,
    PendingShipment,
    PlanStep,
    Reward,
    SpeedMode,
    TaskDefinition,
    WeatherEvent,
)


class EcoLogisticsEnv:
    """Multi-city supply chain environment with carbon tracking + v2 curriculum."""

    def __init__(self, seed: int = 42):
        # ─── v1 state ──────────────────────────────────────────────────────
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
        self._last_demand: Dict[str, float] = {}
        self._last_fulfilled: Dict[str, float] = {}

        # ─── v2 state ──────────────────────────────────────────────────────
        self._curriculum: CurriculumState = CurriculumState()
        self._using_v2_tasks: bool = False
        self._last_grader_result: Optional[GraderResult] = None

        # ─── v2 replanning state ───────────────────────────────────────────
        # current_plan: full plan emitted by agent (covers remaining steps)
        # plan_starting_step: step index at which current_plan starts
        # original_plan_first_emission: first plan ever emitted this episode
        #   (kept so we can score plan_consistency_bonus on revisions)
        self._current_plan: List[Dict[str, Any]] = []
        self._plan_starting_step: int = 0
        self._original_plan_first_emission: List[Dict[str, Any]] = []
        self._plan_revisions_count: int = 0

        # ─── v2 multi-agent state ──────────────────────────────────────────
        # role_plans: per-role plans currently being executed
        # role_recent_actions: last action emitted by each role (for peer obs)
        # role_negotiation_status: per-role current negotiation status
        # last_negotiation_outcomes: list of {step, accepted, roles, bonus}
        self._role_plans: Dict[str, List[Dict[str, Any]]] = {}
        self._role_recent_actions: Dict[str, List[Dict[str, Any]]] = {}
        self._role_negotiation_status: Dict[str, NegotiationStatus] = {}
        self._last_negotiation_outcomes: List[Dict[str, Any]] = []
        self._cumulative_team_carbon_bonus: float = 0.0
        self._cumulative_negotiation_bonus: float = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        task_id: str,
        seed: Optional[int] = None,
        use_v2: bool = False,
    ) -> Observation:
        """Reset the env at a specific task.

        v1 callers pass task_id like before — use_v2 defaults to False, so
        nothing changes.

        v2 callers pass use_v2=True to opt into the 25-step extended task
        variants from TASKS_V2.
        """
        # Pick the task table to use (v1 or v2)
        task_table = TASKS_V2 if use_v2 else TASKS
        if task_id not in task_table:
            raise ValueError(
                f"Unknown task: {task_id}. Available: {list(task_table.keys())}"
            )

        self._using_v2_tasks = use_v2
        self._rng = random.Random(seed if seed is not None else self._seed)
        self._task = task_table[task_id]
        self._inventory = deepcopy(self._task.initial_inventory)
        self._pending = []
        self._weather = WeatherEvent()
        self._step_num = 0
        self._cumulative_profit = 0.0
        self._cumulative_carbon = 0.0
        self._carbon_budget = self._task.carbon_budget
        self._history = []
        self._done = False
        self._last_demand = {}
        self._last_fulfilled = {}
        self._last_grader_result = None

        # v2 replanning reset
        self._current_plan = []
        self._plan_starting_step = 0
        self._original_plan_first_emission = []
        self._plan_revisions_count = 0

        # v2 multi-agent reset
        self._role_plans = {}
        self._role_recent_actions = {}
        self._role_negotiation_status = {}
        self._last_negotiation_outcomes = []
        self._cumulative_team_carbon_bonus = 0.0
        self._cumulative_negotiation_bonus = 0.0

        return self._make_observation()

    def reset_to_curriculum_level(self, seed: Optional[int] = None) -> Observation:
        """v2 helper: reset to whatever level the curriculum is currently at.

        Uses TASKS_V2 (25-step variants). The curriculum object itself
        persists across episodes so it can advance over time.
        """
        task_id = self._curriculum.current_task_id()
        return self.reset(task_id=task_id, seed=seed, use_v2=True)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP  (v1 — unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._task is None:
            raise RuntimeError("Call reset() before step()")
        if self._done:
            raise RuntimeError("Episode is done. Call reset().")

        reward = self._process_step(action)
        self._step_num += 1

        if self._step_num >= self._task.total_steps:
            self._done = True
            # When using curriculum mode, auto-record outcome for advancement
            if self._using_v2_tasks:
                self._record_episode_for_curriculum()

        obs = self._make_observation()

        info: Dict[str, Any] = {
            "step_number": self._step_num,
            "cumulative_profit": round(self._cumulative_profit, 2),
            "cumulative_carbon": round(self._cumulative_carbon, 2),
            "carbon_budget_remaining": round(self._carbon_budget - self._cumulative_carbon, 2),
            "weather_active": self._weather.steps_remaining > 0,
            "pending_shipment_count": len(self._pending),
            # v2-only fields (always present, just zeroed in v1 mode)
            "curriculum_level": self._curriculum.current_level,
            "curriculum_success_rate": round(self._curriculum.success_rate(), 3),
            "using_v2_tasks": self._using_v2_tasks,
        }

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

    # ─────────────────────────────────────────────────────────────────────────
    # CORE SIMULATION (v1 — unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    def _process_step(self, action: Action) -> Reward:
        demand = self._generate_demand()
        self._tick_weather()
        self._deliver_shipments()

        for city in CITIES:
            self._inventory[city] *= (1.0 - DECAY_RATE)

        for city in CITIES:
            self._inventory[city] += RESTOCK_AMOUNT

        shipping_cost, carbon_emitted = self._process_action(action)

        sales_revenue = 0.0
        fulfilled = {}
        for city in CITIES:
            sold = min(self._inventory[city], demand[city])
            sales_revenue += sold * SELL_PRICE
            self._inventory[city] -= sold
            fulfilled[city] = sold

        self._last_demand = demand
        self._last_fulfilled = fulfilled

        storage_fee = sum(max(0, v) for v in self._inventory.values()) * STORAGE_FEE_PER_UNIT

        self._cumulative_carbon += carbon_emitted
        carbon_penalty = carbon_emitted * CARBON_TAX_PER_UNIT

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
                base_demand = 12.0 + self._rng.gauss(0, 2)
                if city == "NYC" and self._step_num >= 7:
                    base = base_demand * 3.0
                else:
                    base = base_demand
            else:  # volatile
                base = 12.0 + self._rng.gauss(0, 6)

            demand[city] = max(0.0, round(base, 1))

        return demand

    def _tick_weather(self):
        if self._weather.steps_remaining > 0:
            self._weather.steps_remaining -= 1
        else:
            chance = 0.15 if (self._task and self._task.demand_profile == "volatile") else 0.05
            if self._rng.random() < chance:
                routes = [("Chicago", "NYC"), ("Seattle", "Chicago"), ("Seattle", "NYC")]
                route = self._rng.choice(routes)
                self._weather = WeatherEvent(
                    affected_route=route,
                    cost_multiplier=5.0,
                    steps_remaining=2,
                )

    def _deliver_shipments(self):
        arrived = [s for s in self._pending if s.steps_remaining <= 0]
        self._pending = [s for s in self._pending if s.steps_remaining > 0]
        for s in arrived:
            self._inventory[s.destination] += s.amount
        for s in self._pending:
            s.steps_remaining -= 1

    def _process_action(self, action: Action) -> Tuple[float, float]:
        if action.ship_amount <= 0:
            return 0.0, 0.0

        if action.origin_city == action.destination_city:
            return 0.0, 0.0

        route = (action.origin_city, action.destination_city)
        if route not in CITY_PAIRS:
            return 0.0, 0.0

        actual = min(action.ship_amount, self._inventory[action.origin_city])
        if actual <= 0:
            return 0.0, 0.0

        info = CITY_PAIRS[route]
        is_air = action.speed_mode == SpeedMode.AIR

        base_cost = info["air_cost"] if is_air else info["rail_cost"]
        lead_time = info["air_steps"] if is_air else info["rail_steps"]
        carbon = info["air_carbon"] if is_air else info["rail_carbon"]

        fwd = route
        rev = (route[1], route[0])
        if self._weather.steps_remaining > 0 and (
            fwd == self._weather.affected_route or rev == self._weather.affected_route
        ):
            base_cost *= self._weather.cost_multiplier

        cost = base_cost * actual
        carbon_total = carbon * actual

        self._inventory[action.origin_city] -= actual

        self._pending.append(PendingShipment(
            origin=action.origin_city,
            destination=action.destination_city,
            amount=actual,
            steps_remaining=lead_time,
            speed_mode=action.speed_mode,
        ))

        return round(cost, 2), round(carbon_total, 2)

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVATION + STATE  (v1 — unchanged)
    # ─────────────────────────────────────────────────────────────────────────

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
            # v2 additions (always present, harmless to v1 callers)
            "curriculum_level": self._curriculum.current_level,
            "curriculum_success_rate": round(self._curriculum.success_rate(), 3),
            "curriculum_episodes_at_level": self._curriculum.episodes_at_level,
            "using_v2_tasks": self._using_v2_tasks,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # GRADERS  (v1 — unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    def grade(self, task_id: Optional[str] = None) -> GraderResult:
        tid = task_id or (self._task.id if self._task else None)
        if tid is None:
            raise ValueError("No task to grade.")

        if tid == "restock_only":
            result = self._grade_restock()
        elif tid == "inventory_balanced":
            result = self._grade_balanced()
        elif tid == "net_zero_profit":
            result = self._grade_net_zero()
        else:
            raise ValueError(f"Unknown task: {tid}")

        self._last_grader_result = result
        return result

    @staticmethod
    def _clamp_score(score: float) -> float:
        return min(0.999, max(0.001, score))

    def _grade_restock(self) -> GraderResult:
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
        if not self._history:
            return GraderResult(task_id="net_zero_profit", score=0.001, feedback="No steps taken.")

        carbon_ok = self._cumulative_carbon <= self._carbon_budget
        profit = self._cumulative_profit

        max_expected_profit = self._task.total_steps * len(CITIES) * 15.0 * SELL_PRICE
        profit_score = max(0.0, profit) / max_expected_profit if max_expected_profit > 0 else 0.0
        profit_score = min(1.0, profit_score)

        if not carbon_ok:
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

    # ─────────────────────────────────────────────────────────────────────────
    # v2 CURRICULUM API
    # ─────────────────────────────────────────────────────────────────────────

    def get_curriculum_state(self) -> CurriculumState:
        """Return a copy of the current curriculum state (for inspection)."""
        return self._curriculum.model_copy()

    def reset_curriculum(self) -> None:
        """Manually reset the curriculum back to level 0. Useful for fresh runs."""
        self._curriculum = CurriculumState()

    def force_curriculum_level(self, level: int) -> None:
        """Manually set the curriculum level. Useful for evaluation at a fixed
        level without going through the auto-advance schedule.
        """
        if level not in (0, 1, 2):
            raise ValueError(f"Curriculum level must be 0, 1, or 2 — got {level}")
        self._curriculum = CurriculumState(current_level=level)

    def _record_episode_for_curriculum(self) -> bool:
        """Called automatically at episode end when using_v2_tasks=True.
        Grades the just-completed episode and advances curriculum if threshold met.

        Returns True if level advanced.
        """
        if not self._using_v2_tasks or self._task is None:
            return False

        # Grade the episode (also caches in self._last_grader_result)
        try:
            grader = self.grade()
            success = grader.score >= self._task.success_threshold
        except Exception:
            success = False

        return self._curriculum.record_episode(success)

    # ─────────────────────────────────────────────────────────────────────────
    # v2 RECEDING-HORIZON REPLANNING API
    # ─────────────────────────────────────────────────────────────────────────
    #
    # Old way (v1): agent emits 1 action → env runs 1 step. Repeat 10 times.
    # New way (v2): agent emits a 10-step plan → env runs REPLAN_INTERVAL_STEPS
    #               (=4) steps from it → asks "REVISE PLAN?" → agent can either
    #               keep going with the same plan, OR submit new plan for the
    #               remaining steps.
    #
    # Method: submit_plan(plan)              ← agent gives env a plan
    #         run_plan_chunk()               ← env runs next chunk
    #         needs_replan() / step_number   ← agent checks if revise time
    #         submit_revised_plan(plan)      ← agent gives revised plan

    def submit_plan(self, plan: AgentPlan) -> None:
        """Agent submits initial 10-step plan. Stored, not run yet."""
        if self._task is None:
            raise RuntimeError("Call reset() before submit_plan()")
        if self._done:
            raise RuntimeError("Episode done. Call reset().")

        steps_dict = [s.model_dump() for s in plan.steps]
        self._current_plan = steps_dict
        self._plan_starting_step = self._step_num
        if not self._original_plan_first_emission:
            # first plan of episode - keep for consistency scoring later
            self._original_plan_first_emission = list(steps_dict)
        self._plan_revisions_count = 0

    def submit_revised_plan(self, plan: AgentPlan) -> None:
        """Agent revises remaining plan mid-episode.

        plan.steps should cover ONLY the steps from current step_num to end.
        Bonus +5 awarded if revised plan still ships similar amounts to
        original plan for those positions (plan_consistency_bonus).
        """
        if self._task is None:
            raise RuntimeError("Call reset() before submit_revised_plan()")
        if self._done:
            raise RuntimeError("Episode done.")

        new_steps = [s.model_dump() for s in plan.steps]
        self._current_plan = new_steps
        self._plan_starting_step = self._step_num
        self._plan_revisions_count += 1

    def needs_replan(self) -> bool:
        """True when agent should consider revising plan (every 4 steps)."""
        if self._step_num == 0:
            return False
        return (self._step_num % REPLAN_INTERVAL_STEPS == 0) and not self._done

    def run_plan_chunk(
        self, max_chunk_steps: Optional[int] = None
    ) -> Tuple[Observation, List[Reward], bool, Dict[str, Any]]:
        """Run env steps using actions from current_plan.

        Stops when:
        - chunk size reached (default REPLAN_INTERVAL_STEPS = 4 steps)
        - episode done
        - plan exhausted

        Returns (last_obs, list_of_rewards, done, info).
        Info includes 'revise_plan_flag' = True if env wants new plan.
        """
        if self._task is None:
            raise RuntimeError("Call reset() before run_plan_chunk()")
        if self._done:
            raise RuntimeError("Episode done.")
        if not self._current_plan:
            raise RuntimeError("No plan submitted. Call submit_plan() first.")

        chunk_size = max_chunk_steps if max_chunk_steps is not None else REPLAN_INTERVAL_STEPS

        rewards_collected: List[Reward] = []
        consistency_bonus_total = 0.0
        last_obs = None
        steps_run = 0

        for _ in range(chunk_size):
            if self._done:
                break

            # index into current plan based on offset from plan_starting_step
            plan_idx = self._step_num - self._plan_starting_step
            if plan_idx >= len(self._current_plan):
                # Plan exhausted but episode still going - emit no-op
                action_dict = {
                    "ship_amount": 0.0,
                    "origin_city": "Seattle",
                    "destination_city": "Chicago",
                    "speed_mode": "Rail",
                }
            else:
                action_dict = self._current_plan[plan_idx]

            # Build Action object (strip extra v2 fields if present)
            try:
                action = Action(
                    ship_amount=action_dict.get("ship_amount", 0.0),
                    origin_city=action_dict.get("origin_city", "Seattle"),
                    destination_city=action_dict.get("destination_city", "Chicago"),
                    speed_mode=SpeedMode(action_dict.get("speed_mode", "Rail")),
                )
            except Exception:
                # Bad plan step - emit no-op
                action = Action(
                    ship_amount=0.0,
                    origin_city="Seattle",
                    destination_city="Chicago",
                    speed_mode=SpeedMode.RAIL,
                )

            # Plan consistency bonus: if this is a post-revision step AND the
            # action at this index in the revised plan still ships similar
            # amount to the original plan's action for the same absolute step,
            # award +PLAN_CONSISTENCY_BONUS.
            if self._plan_revisions_count > 0 and self._original_plan_first_emission:
                abs_step = self._step_num
                if abs_step < len(self._original_plan_first_emission):
                    orig = self._original_plan_first_emission[abs_step]
                    revised_ship = action_dict.get("ship_amount", 0.0)
                    orig_ship = orig.get("ship_amount", 0.0)
                    # consistent if within 30% of original ship_amount
                    if max(orig_ship, 1.0) > 0:
                        ratio = abs(revised_ship - orig_ship) / max(orig_ship, 1.0)
                        if ratio <= 0.3:
                            consistency_bonus_total += PLAN_CONSISTENCY_BONUS

            obs, reward, done, info = self.step(action)
            # Mutate reward in place to add the consistency bonus on the LAST
            # step of the chunk (so bonus is visible per-chunk, not stale)
            rewards_collected.append(reward)
            last_obs = obs
            steps_run += 1

            if done:
                break

        # apply accumulated consistency bonus to last reward
        if consistency_bonus_total > 0 and rewards_collected:
            final = rewards_collected[-1]
            final.plan_consistency_bonus = round(consistency_bonus_total, 2)
            final.total = round(final.total + consistency_bonus_total, 2)
            self._cumulative_profit += consistency_bonus_total

        chunk_info = {
            "steps_run_in_chunk": steps_run,
            "step_number": self._step_num,
            "done": self._done,
            "needs_replan": self.needs_replan(),
            "revise_plan_flag": self.needs_replan(),
            "plan_revisions_so_far": self._plan_revisions_count,
            "consistency_bonus_in_chunk": round(consistency_bonus_total, 2),
            "cumulative_profit": round(self._cumulative_profit, 2),
            "cumulative_carbon": round(self._cumulative_carbon, 2),
        }

        if last_obs is None:
            last_obs = self._make_observation()

        return last_obs, rewards_collected, self._done, chunk_info

    def get_replanning_state(self) -> Dict[str, Any]:
        """Inspect current replanning state (for debugging / observability)."""
        return {
            "current_plan_length": len(self._current_plan),
            "plan_starting_step": self._plan_starting_step,
            "step_number": self._step_num,
            "plan_revisions_count": self._plan_revisions_count,
            "needs_replan": self.needs_replan(),
            "original_plan_length": len(self._original_plan_first_emission),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # v2 MULTI-AGENT API
    # ─────────────────────────────────────────────────────────────────────────
    #
    # 3 roles: seattle_mgr, chicago_router, nyc_carbon
    # Each role emits its own plan. Env runs them together per-step.
    # Negotiation: roleA proposes shared shipment to roleB. If both roles'
    #   plans match the offer (same route, mode, similar amount), the
    #   negotiation is ACCEPTED → both roles get +NEGOTIATION_SUCCESS_BONUS.
    # Team carbon bonus: at episode end, if total carbon stays under cap by
    #   margin > 30%, +TEAM_CARBON_BONUS_MAX is split across all roles.

    def submit_multiagent_plans(
        self, plans: Dict[str, AgentPlan]
    ) -> None:
        """Submit one plan per role. plans: {'seattle_mgr': AgentPlan, ...}.

        Each plan must have steps covering remaining episode length.
        Negotiation proposals on plans are inspected at chunk run time.
        """
        if self._task is None:
            raise RuntimeError("Call reset() first")
        if self._done:
            raise RuntimeError("Episode done")

        valid_roles = {AgentRole.SEATTLE_MGR.value, AgentRole.CHICAGO_ROUTER.value, AgentRole.NYC_CARBON.value}
        for role_str in plans:
            if role_str not in valid_roles:
                raise ValueError(f"Invalid role '{role_str}'. Valid: {valid_roles}")

        for role_str, plan in plans.items():
            self._role_plans[role_str] = [s.model_dump() for s in plan.steps]
            # Store negotiation proposal at plan level
            if plan.negotiation_proposal:
                self._role_negotiation_status[role_str] = NegotiationStatus.PROPOSED
            else:
                self._role_negotiation_status[role_str] = NegotiationStatus.NONE
            self._role_recent_actions.setdefault(role_str, [])

        # Run negotiation matching pass: if 2 roles propose to each other
        # with matching offers, both get accepted + bonus
        self._resolve_negotiations(plans)

    def _resolve_negotiations(self, plans: Dict[str, AgentPlan]) -> None:
        """Match negotiation proposals across roles.

        Rule: if roleA.proposal_to == roleB AND roleB.proposal_to == roleA AND
        their offers describe the same route + mode (regardless of amount),
        both roles get NEGOTIATION_SUCCESS_BONUS. Otherwise no bonus.

        Malformed proposals (missing fields, invalid roles) → INVALID penalty.
        """
        accepted_pairs = []
        invalid_count = 0

        for role_a, plan_a in plans.items():
            prop_a = plan_a.negotiation_proposal
            if not prop_a:
                continue

            # Validate proposal schema
            target = prop_a.get("proposal_to")
            offer = prop_a.get("offer")
            if not target or not offer or not isinstance(offer, dict):
                invalid_count += 1
                self._role_negotiation_status[role_a] = NegotiationStatus.REJECTED
                continue
            if target not in plans:
                # proposing to absent role
                invalid_count += 1
                self._role_negotiation_status[role_a] = NegotiationStatus.REJECTED
                continue

            # Look at target's proposal
            plan_b = plans[target]
            prop_b = plan_b.negotiation_proposal
            if not prop_b:
                continue  # target didn't propose back; one-sided
            if prop_b.get("proposal_to") != role_a:
                continue  # they proposed to someone else

            # Check offers compatible: same route + same mode
            offer_b = prop_b.get("offer", {})
            if (
                offer.get("route") == offer_b.get("route")
                and offer.get("mode") == offer_b.get("mode")
                and offer.get("route") is not None
            ):
                pair_key = tuple(sorted([role_a, target]))
                if pair_key not in accepted_pairs:
                    accepted_pairs.append(pair_key)
                    self._role_negotiation_status[role_a] = NegotiationStatus.ACCEPTED
                    self._role_negotiation_status[target] = NegotiationStatus.ACCEPTED

        # Award bonuses for accepted pairs
        bonus_total = len(accepted_pairs) * NEGOTIATION_SUCCESS_BONUS * 2  # both roles in each pair
        self._cumulative_negotiation_bonus += bonus_total
        self._cumulative_profit += bonus_total

        # Apply invalid penalty
        if invalid_count > 0:
            penalty_total = invalid_count * INVALID_NEGOTIATION_PENALTY
            self._cumulative_profit += penalty_total

        # Record outcome for telemetry
        self._last_negotiation_outcomes.append({
            "step": self._step_num,
            "accepted_pairs": [list(p) for p in accepted_pairs],
            "invalid_count": invalid_count,
            "bonus_awarded": bonus_total,
            "penalty_applied": invalid_count * INVALID_NEGOTIATION_PENALTY,
        })

    def run_multiagent_chunk(
        self, max_chunk_steps: Optional[int] = None
    ) -> Tuple[
        Dict[str, MultiAgentObservation],
        List[Reward],
        bool,
        Dict[str, Any],
    ]:
        """Run a chunk of multi-agent steps.

        Per step: each role's plan contributes one action. Env executes them
        in role order (seattle_mgr → chicago_router → nyc_carbon). Reward is
        computed once per step (env-level), then bonuses added.

        Returns ({role: MultiAgentObservation}, list_of_step_rewards, done, info).
        """
        if self._task is None:
            raise RuntimeError("Call reset() first")
        if self._done:
            raise RuntimeError("Episode done")
        if not self._role_plans:
            raise RuntimeError("No multi-agent plans submitted. Call submit_multiagent_plans() first.")

        chunk_size = max_chunk_steps if max_chunk_steps is not None else REPLAN_INTERVAL_STEPS
        rewards_collected: List[Reward] = []
        steps_run = 0

        for _ in range(chunk_size):
            if self._done:
                break

            # Each role contributes one action this step
            step_actions: Dict[str, Action] = {}
            for role_str, plan in self._role_plans.items():
                # Plan is anchored at plan_starting_step; index relative to current step
                plan_idx = self._step_num - self._plan_starting_step
                if plan_idx < 0 or plan_idx >= len(plan):
                    # plan exhausted — no-op for this role
                    action = Action(
                        ship_amount=0.0,
                        origin_city="Seattle",
                        destination_city="Chicago",
                        speed_mode=SpeedMode.RAIL,
                        role=AgentRole(role_str),
                    )
                else:
                    a_dict = plan[plan_idx]
                    try:
                        action = Action(
                            ship_amount=a_dict.get("ship_amount", 0.0),
                            origin_city=a_dict.get("origin_city", "Seattle"),
                            destination_city=a_dict.get("destination_city", "Chicago"),
                            speed_mode=SpeedMode(a_dict.get("speed_mode", "Rail")),
                            role=AgentRole(role_str),
                        )
                    except Exception:
                        action = Action(
                            ship_amount=0.0,
                            origin_city="Seattle",
                            destination_city="Chicago",
                            speed_mode=SpeedMode.RAIL,
                            role=AgentRole(role_str),
                        )
                step_actions[role_str] = action

            # Compute the per-step reward by running each role's action.
            # We accumulate sales_revenue, costs, carbon across all roles
            # but only generate demand and apply storage/decay ONCE per step.
            step_reward = self._process_multiagent_step(step_actions)
            self._step_num += 1

            rewards_collected.append(step_reward)
            steps_run += 1

            # Track recent actions for peer observation
            for role_str, action in step_actions.items():
                if role_str not in self._role_recent_actions:
                    self._role_recent_actions[role_str] = []
                self._role_recent_actions[role_str].append({
                    "role": role_str,
                    "step": self._step_num,
                    "action": action.model_dump(),
                })
                # cap recent history to last 5
                if len(self._role_recent_actions[role_str]) > 5:
                    self._role_recent_actions[role_str] = self._role_recent_actions[role_str][-5:]

            if self._step_num >= self._task.total_steps:
                self._done = True
                if self._using_v2_tasks:
                    self._record_episode_for_curriculum()
                # Compute team carbon bonus at episode end
                self._compute_team_carbon_bonus(rewards_collected)
                break

        # Build per-role observations
        role_obs: Dict[str, MultiAgentObservation] = {}
        base_obs = self._make_observation()
        for role_str in self._role_plans.keys():
            other_actions = []
            for other_role, recent in self._role_recent_actions.items():
                if other_role != role_str and recent:
                    other_actions.extend(recent[-2:])  # last 2 actions from each peer
            role_obs[role_str] = MultiAgentObservation(
                base=base_obs,
                role=AgentRole(role_str),
                other_agents_recent_actions=other_actions,
                revise_plan_flag=self.needs_replan(),
                original_plan_remaining=[],
                curriculum_level=self._curriculum.current_level,
                curriculum_recent_success_rate=self._curriculum.success_rate(),
                open_proposals=[],
                last_negotiation_status=self._role_negotiation_status.get(role_str, NegotiationStatus.NONE),
            )

        info = {
            "steps_run_in_chunk": steps_run,
            "step_number": self._step_num,
            "done": self._done,
            "needs_replan": self.needs_replan(),
            "revise_plan_flag": self.needs_replan(),
            "cumulative_profit": round(self._cumulative_profit, 2),
            "cumulative_carbon": round(self._cumulative_carbon, 2),
            "cumulative_negotiation_bonus": round(self._cumulative_negotiation_bonus, 2),
            "cumulative_team_carbon_bonus": round(self._cumulative_team_carbon_bonus, 2),
            "negotiation_outcomes": self._last_negotiation_outcomes[-1] if self._last_negotiation_outcomes else None,
            "role_negotiation_status": {r: s.value for r, s in self._role_negotiation_status.items()},
        }

        return role_obs, rewards_collected, self._done, info

    def _process_multiagent_step(self, step_actions: Dict[str, Action]) -> Reward:
        """Run one step where multiple roles ship in the same tick.

        Differs from _process_step: we run each role's shipping action,
        accumulate costs/carbon, then do demand fulfillment ONCE.
        """
        # 1. Demand
        demand = self._generate_demand()

        # 2. Weather tick
        self._tick_weather()

        # 3. Deliver pending
        self._deliver_shipments()

        # 4. Decay
        for city in CITIES:
            self._inventory[city] *= (1.0 - DECAY_RATE)

        # 5. Restock
        for city in CITIES:
            self._inventory[city] += RESTOCK_AMOUNT

        # 6. Each role ships (in role order)
        total_ship_cost = 0.0
        total_carbon = 0.0
        for role_str in [AgentRole.SEATTLE_MGR.value, AgentRole.CHICAGO_ROUTER.value, AgentRole.NYC_CARBON.value]:
            if role_str in step_actions:
                cost, carbon = self._process_action(step_actions[role_str])
                total_ship_cost += cost
                total_carbon += carbon

        # 7. Fulfill demand
        sales_revenue = 0.0
        fulfilled = {}
        for city in CITIES:
            sold = min(self._inventory[city], demand[city])
            sales_revenue += sold * SELL_PRICE
            self._inventory[city] -= sold
            fulfilled[city] = sold
        self._last_demand = demand
        self._last_fulfilled = fulfilled

        # 8. Storage fee
        storage_fee = sum(max(0, v) for v in self._inventory.values()) * STORAGE_FEE_PER_UNIT

        # 9. Carbon
        self._cumulative_carbon += total_carbon
        carbon_penalty = total_carbon * CARBON_TAX_PER_UNIT

        # 10. Healthy stock bonus
        healthy = all(self._inventory[c] >= 20.0 for c in CITIES)
        bonus = HEALTHY_STOCK_BONUS if healthy else 0.0

        total = sales_revenue - total_ship_cost - carbon_penalty - storage_fee + bonus
        self._cumulative_profit += total

        return Reward(
            sales_revenue=round(sales_revenue, 2),
            shipping_cost=round(total_ship_cost, 2),
            carbon_penalty=round(carbon_penalty, 2),
            storage_fee=round(storage_fee, 2),
            healthy_stock_bonus=round(bonus, 2),
            total=round(total, 2),
        )

    def _compute_team_carbon_bonus(self, rewards: List[Reward]) -> None:
        """End-of-episode: if total carbon < 70% of budget, award team bonus."""
        if self._task is None or self._carbon_budget <= 0:
            return
        carbon_used_ratio = self._cumulative_carbon / self._carbon_budget
        if carbon_used_ratio < 0.70:
            # Slack proportional bonus: more slack = more bonus (capped at MAX)
            slack = max(0.0, 1.0 - carbon_used_ratio)
            bonus = min(TEAM_CARBON_BONUS_MAX, slack * TEAM_CARBON_BONUS_MAX)
            self._cumulative_team_carbon_bonus = round(bonus, 2)
            self._cumulative_profit += bonus
            # Apply to last reward of the episode
            if rewards:
                rewards[-1].team_carbon_bonus = round(bonus, 2)
                rewards[-1].total = round(rewards[-1].total + bonus, 2)

    def get_multiagent_state(self) -> Dict[str, Any]:
        """Inspect multi-agent state."""
        return {
            "active_roles": list(self._role_plans.keys()),
            "role_plan_lengths": {r: len(p) for r, p in self._role_plans.items()},
            "role_negotiation_status": {r: s.value for r, s in self._role_negotiation_status.items()},
            "negotiation_outcomes_so_far": len(self._last_negotiation_outcomes),
            "cumulative_negotiation_bonus": round(self._cumulative_negotiation_bonus, 2),
            "cumulative_team_carbon_bonus": round(self._cumulative_team_carbon_bonus, 2),
        }