"""
Eco-Logistics — Automated Test Suite
Run: python test_env.py
"""

import sys
import json
from env import EcoLogisticsEnv
from models import Action, SpeedMode, TASKS, Observation, Reward, GraderResult, CITIES


def test(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        test.failures += 1
    test.total += 1

test.failures = 0
test.total = 0


def run_tests():
    print("=" * 60)
    print("Eco-Logistics — Test Suite")
    print("=" * 60)

    # ── 1. Initialization ────────────────────────────────────────
    print("\n1. Environment Initialization")
    env = EcoLogisticsEnv(seed=42)
    test("Creates without error", env is not None)

    # ── 2. Reset ─────────────────────────────────────────────────
    print("\n2. Reset Behavior")
    for tid in TASKS:
        obs = env.reset(tid, seed=42)
        test(f"reset({tid}) returns Observation", isinstance(obs, Observation))
        test(f"  step_number is 0", obs.step_number == 0)
        test(f"  all cities in inventory", all(c in obs.current_inventory for c in CITIES))
        test(f"  all cities in demand", all(c in obs.current_demand for c in CITIES))
        test(f"  carbon_credit_balance > 0", obs.carbon_credit_balance > 0)

    # Cannot step without reset on fresh env
    env2 = EcoLogisticsEnv()
    try:
        env2.step(Action(ship_amount=0, origin_city="Seattle", destination_city="Chicago", speed_mode=SpeedMode.RAIL))
        test("Step before reset raises error", False)
    except RuntimeError:
        test("Step before reset raises RuntimeError", True)

    # ── 3. Step returns 4-tuple ──────────────────────────────────
    print("\n3. Step Behavior")
    env.reset("restock_only", seed=42)
    result = env.step(Action(ship_amount=0, origin_city="Seattle", destination_city="Chicago", speed_mode=SpeedMode.RAIL))
    test("step() returns 4 values", len(result) == 4)
    obs, reward, done, info = result
    test("  obs is Observation", isinstance(obs, Observation))
    test("  reward is Reward", isinstance(reward, Reward))
    test("  done is bool", isinstance(done, bool))
    test("  info is dict", isinstance(info, dict))
    test("  info has step_number", "step_number" in info)
    test("  info has cumulative_profit", "cumulative_profit" in info)
    test("  info has cumulative_carbon", "cumulative_carbon" in info)

    # ── 4. Reward components ─────────────────────────────────────
    print("\n4. Reward Structure")
    env.reset("net_zero_profit", seed=42)
    _, reward, _, _ = env.step(Action(ship_amount=10, origin_city="Seattle", destination_city="NYC", speed_mode=SpeedMode.AIR))
    test("reward.sales_revenue exists", hasattr(reward, 'sales_revenue'))
    test("reward.shipping_cost exists", hasattr(reward, 'shipping_cost'))
    test("reward.carbon_penalty exists", hasattr(reward, 'carbon_penalty'))
    test("reward.storage_fee exists", hasattr(reward, 'storage_fee'))
    test("reward.healthy_stock_bonus exists", hasattr(reward, 'healthy_stock_bonus'))
    test("reward.total exists", hasattr(reward, 'total'))
    test("shipping_cost > 0 for air shipment", reward.shipping_cost > 0)
    test("carbon_penalty > 0 for air shipment", reward.carbon_penalty > 0)

    # No-op has no shipping cost
    env.reset("restock_only", seed=42)
    _, reward, _, _ = env.step(Action(ship_amount=0, origin_city="Seattle", destination_city="Chicago", speed_mode=SpeedMode.RAIL))
    test("no-op has zero shipping_cost", reward.shipping_cost == 0)
    test("no-op has zero carbon_penalty", reward.carbon_penalty == 0)

    # ── 5. Episode boundaries ────────────────────────────────────
    print("\n5. Episode Boundaries")
    env.reset("restock_only", seed=42)  # 10 steps
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(Action(ship_amount=0, origin_city="Seattle", destination_city="Chicago", speed_mode=SpeedMode.RAIL))
        steps += 1
    test("restock_only runs exactly 10 steps", steps == 10)

    try:
        env.step(Action(ship_amount=0, origin_city="Seattle", destination_city="Chicago", speed_mode=SpeedMode.RAIL))
        test("Step after done raises error", False)
    except RuntimeError:
        test("Step after done raises RuntimeError", True)

    # Can reset after done
    obs = env.reset("restock_only", seed=42)
    test("Reset after done works", obs.step_number == 0)

    # ── 6. Determinism ───────────────────────────────────────────
    print("\n6. Determinism (same seed = same results)")
    for tid in TASKS:
        rewards1, rewards2 = [], []
        for rewards_list, seed in [(rewards1, 42), (rewards2, 42)]:
            e = EcoLogisticsEnv(seed=seed)
            obs = e.reset(tid, seed=seed)
            done = False
            while not done:
                obs, r, done, _ = e.step(Action(ship_amount=5, origin_city="Seattle", destination_city="NYC", speed_mode=SpeedMode.RAIL))
                rewards_list.append(r.total)
        test(f"  {tid}: same seed produces identical rewards", rewards1 == rewards2)

    # Different seeds produce different results
    e1 = EcoLogisticsEnv(seed=42)
    e2 = EcoLogisticsEnv(seed=99)
    o1 = e1.reset("net_zero_profit", seed=42)
    o2 = e2.reset("net_zero_profit", seed=99)
    test("Different seeds produce different demand", o1.current_demand != o2.current_demand)

    # ── 7. Grader scores strictly in (0, 1) ──────────────────────
    print("\n7. Grader Scores — strictly in (0.0, 1.0)")
    strategies = [
        ("no-op", lambda: Action(ship_amount=0, origin_city="Seattle", destination_city="Chicago", speed_mode=SpeedMode.RAIL)),
        ("air-heavy", lambda: Action(ship_amount=15, origin_city="Seattle", destination_city="NYC", speed_mode=SpeedMode.AIR)),
        ("rail-light", lambda: Action(ship_amount=5, origin_city="Chicago", destination_city="NYC", speed_mode=SpeedMode.RAIL)),
    ]
    for tid in TASKS:
        for sname, sfn in strategies:
            e = EcoLogisticsEnv(seed=42)
            obs = e.reset(tid, seed=42)
            done = False
            while not done:
                obs, r, done, _ = e.step(sfn())
            g = e.grade()
            test(f"  {tid} + {sname}: {g.score:.4f} in (0,1)", 0.0 < g.score < 1.0, f"score={g.score}")

    # Grader variance (not always same score)
    print("\n8. Grader Variance (different strategies → different scores)")
    for tid in TASKS:
        scores = set()
        for _, sfn in strategies:
            e = EcoLogisticsEnv(seed=42)
            obs = e.reset(tid, seed=42)
            done = False
            while not done:
                obs, r, done, _ = e.step(sfn())
            scores.add(round(e.grade().score, 4))
        test(f"  {tid}: {len(scores)} unique scores", len(scores) > 1, f"scores={scores}")

    # ── 9. State ─────────────────────────────────────────────────
    print("\n9. State Method")
    env.reset("restock_only", seed=42)
    env.step(Action(ship_amount=0, origin_city="Seattle", destination_city="Chicago", speed_mode=SpeedMode.RAIL))
    st = env.state()
    test("state() returns dict", isinstance(st, dict))
    for key in ["task_id", "step_number", "done", "inventory", "cumulative_profit", "cumulative_carbon"]:
        test(f"  state has '{key}'", key in st)

    # ── 10. Weather events ───────────────────────────────────────
    print("\n10. Weather & Shipping Mechanics")
    # Run volatile task with many steps to trigger weather
    env.reset("net_zero_profit", seed=42)
    weather_seen = False
    for _ in range(20):
        obs, _, done, _ = env.step(Action(ship_amount=0, origin_city="Seattle", destination_city="Chicago", speed_mode=SpeedMode.RAIL))
        if obs.weather_alert:
            weather_seen = True
            break
        if done:
            break
    test("Weather event triggers in volatile mode", weather_seen)

    # Shipping reduces origin inventory
    env.reset("restock_only", seed=42)
    inv_before = env._inventory["Seattle"]
    env.step(Action(ship_amount=10, origin_city="Seattle", destination_city="NYC", speed_mode=SpeedMode.RAIL))
    # After production (+20) and decay, inventory should be less than before + 20
    test("Shipping deducts from origin inventory", env._inventory["Seattle"] < inv_before + 20)

    # ── 11. Dense reward check ───────────────────────────────────
    print("\n11. Dense Reward Signal")
    env.reset("net_zero_profit", seed=42)
    rewards = []
    done = False
    while not done:
        _, r, done, _ = env.step(Action(ship_amount=3, origin_city="Seattle", destination_city="NYC", speed_mode=SpeedMode.RAIL))
        rewards.append(r.total)
    unique = len(set(round(r, 2) for r in rewards))
    test(f"Reward varies across steps: {unique} unique values", unique >= 10, f"out of {len(rewards)} steps")

    # ── 12. JSON serialization ───────────────────────────────────
    print("\n12. JSON Serialization")
    env.reset("restock_only", seed=42)
    obs, reward, done, info = env.step(Action(ship_amount=0, origin_city="Seattle", destination_city="Chicago", speed_mode=SpeedMode.RAIL))
    try:
        json.dumps(obs.model_dump(), default=str)
        test("Observation serializes to JSON", True)
    except Exception as e:
        test("Observation serializes to JSON", False, str(e))
    try:
        json.dumps(reward.model_dump())
        test("Reward serializes to JSON", True)
    except Exception as e:
        test("Reward serializes to JSON", False, str(e))

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    passed = test.total - test.failures
    print(f"Results: {passed}/{test.total} passed, {test.failures} failed")
    if test.failures == 0:
        print("ALL TESTS PASSED")
    print(f"{'='*60}")
    return test.failures == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)