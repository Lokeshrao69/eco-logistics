"""
Eco-Logistics — Heuristic Baseline Agent

Used by the /baseline FastAPI endpoint. Deterministic rule-based policy.
For LLM inference, see inference.py.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from env import EcoLogisticsEnv
from models import CITIES, Action, SpeedMode


def _pick_heuristic_action(obs_dict: Dict[str, Any]) -> Action:
    """
    Simple rule:
    - Find the city with the most stock and the city with the least stock.
    - If the gap > 15, ship from surplus to deficit via Rail.
    - Otherwise, no-op.
    """
    inv = obs_dict["current_inventory"]
    cities_sorted = sorted(CITIES, key=lambda c: inv[c])
    min_city = cities_sorted[0]
    max_city = cities_sorted[-1]

    gap = inv[max_city] - inv[min_city]

    if gap > 15 and inv[max_city] > 30:
        amount = min(gap * 0.4, inv[max_city] * 0.3)
        weather = obs_dict.get("weather_alert")
        route_str = f"{max_city}\u2192{min_city}"
        if weather and route_str in weather:
            mid_city = [c for c in CITIES if c != max_city and c != min_city][0]
            alt_route = f"{max_city}\u2192{mid_city}"
            if weather and alt_route in weather:
                return Action(ship_amount=0, origin_city="Seattle", destination_city="Chicago", speed_mode=SpeedMode.RAIL)
            return Action(
                ship_amount=round(amount, 1),
                origin_city=max_city,
                destination_city=mid_city,
                speed_mode=SpeedMode.RAIL,
            )
        return Action(
            ship_amount=round(amount, 1),
            origin_city=max_city,
            destination_city=min_city,
            speed_mode=SpeedMode.RAIL,
        )

    return Action(ship_amount=0, origin_city="Seattle", destination_city="Chicago", speed_mode=SpeedMode.RAIL)


def run_heuristic_baseline(task_id: str = "restock_only", seed: Optional[int] = 42) -> Dict[str, Any]:
    """Run the heuristic agent through a full episode and return grade + logs."""
    env = EcoLogisticsEnv(seed=seed)
    obs = env.reset(task_id=task_id, seed=seed)
    steps_log = []

    done = False
    step_num = 0
    while not done:
        obs_dict = obs.model_dump()
        action = _pick_heuristic_action(obs_dict)
        obs, reward, done, info = env.step(action)
        steps_log.append({
            "step": step_num,
            "action": action.model_dump(),
            "reward_total": reward.total,
            "inventory": obs.current_inventory,
        })
        step_num += 1

    grade = env.grade()
    return {
        "task_id": task_id,
        "grade": grade.model_dump(),
        "steps": steps_log,
    }


if __name__ == "__main__":
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else "restock_only"
    result = run_heuristic_baseline(task_id=task)
    print(f"Task: {result['task_id']}")
    print(f"Score: {result['grade']['score']}")
    print(f"Feedback: {result['grade']['feedback']}")
