"""
Eco-Logistics: Multi-City Supply Chain Optimizer — LLM Inference Script
=======================================================================

MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

Usage:
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="gpt-4o"
    export HF_TOKEN="hf_..."
    python inference.py

Participants must use OpenAI Client for all LLM calls using above variables.
"""

import json
import os
import re
import sys
import textwrap
import time

from openai import OpenAI

from env import EcoLogisticsEnv
from models import TASKS, Action, SpeedMode

# ═══════════════════════════════════════════════════════════════════════════
# Configuration — from environment variables
# ═══════════════════════════════════════════════════════════════════════════

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# Inference parameters
SEED = 42
TEMPERATURE = 0.0
MAX_TOKENS = 300
MAX_HISTORY = 20          # max conversation messages before truncation
DEBUG = True

# Fallback when LLM fails to produce valid JSON
FALLBACK_ACTION = {
    "ship_amount": 0,
    "origin_city": "Seattle",
    "destination_city": "Chicago",
    "speed_mode": "Rail",
}

# Task IDs to run (all three)
TASK_IDS = ["restock_only", "inventory_balanced", "net_zero_profit"]

# Valid values for sanitization
VALID_CITIES = ["Seattle", "Chicago", "NYC"]
VALID_MODES = ["Air", "Rail"]

# Regex to extract JSON from LLM response
JSON_PATTERN = re.compile(r"\{[^{}]*\}", re.DOTALL)


# ═══════════════════════════════════════════════════════════════════════════
# System Prompt — task-specific, injected per episode
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert AI supply chain manager controlling inventory across
    3 warehouses: Seattle, Chicago, NYC.

    ENVIRONMENT MECHANICS:
    - Each city produces +20 units/step and decays 2%/step.
    - Demand is fulfilled from local inventory at $10/unit revenue.
    - Storage costs $0.50/unit on all remaining inventory.
    - Carbon tax: $1.50 per carbon unit emitted.
    - Healthy stock bonus: +$0.10 if ALL cities have >= 20 units.

    SHIPPING ROUTES (cost / lead-time / carbon, per unit shipped):
      Seattle <-> Chicago : Rail $3/3steps/2C   Air $8/1step/8C
      Chicago <-> NYC     : Rail $2/2steps/1.5C Air $6/1step/6C
      Seattle <-> NYC     : Rail $5/3steps/3.5C Air $12/1step/12C

    WEATHER: Routes randomly become 5x more expensive for 2 steps.

    YOUR CURRENT TASK: {task_description}

    Reply with ONLY a valid JSON object, no markdown, no explanation:
    {{"ship_amount": <float>, "origin_city": "<city>", "destination_city": "<city>", "speed_mode": "<Air|Rail>"}}

    STRATEGY:
    - Restock Only: mostly no-op (ship_amount=0), production covers demand.
    - Inventory Balanced: redistribute from surplus to deficit via Rail.
    - Net-Zero Profit: carbon budget is VERY tight. Prefer no-op. Only ship
      via Rail when absolutely necessary. Every shipment burns carbon.
    - If a weather alert is active on a route, AVOID that route entirely.
    - ship_amount=0 is always a safe action.
""")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def parse_action(raw_text: str) -> dict:
    """
    Extract and sanitize a JSON action from the LLM's raw response.
    Falls back to FALLBACK_ACTION on any failure.
    """
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try to find a JSON object
    match = JSON_PATTERN.search(text)
    if not match:
        if DEBUG:
            print(f"    [PARSE] No JSON found in: {text[:80]}...")
        return FALLBACK_ACTION.copy()

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        if DEBUG:
            print(f"    [PARSE] Invalid JSON: {match.group()[:80]}...")
        return FALLBACK_ACTION.copy()

    # Sanitize fields
    action = {
        "ship_amount": max(0.0, float(data.get("ship_amount", 0))),
        "origin_city": data.get("origin_city", "Seattle"),
        "destination_city": data.get("destination_city", "Chicago"),
        "speed_mode": data.get("speed_mode", "Rail"),
    }

    if action["origin_city"] not in VALID_CITIES:
        action["origin_city"] = "Seattle"
    if action["destination_city"] not in VALID_CITIES:
        action["destination_city"] = "Chicago"
    if action["speed_mode"] not in VALID_MODES:
        action["speed_mode"] = "Rail"
    if action["origin_city"] == action["destination_city"]:
        action["ship_amount"] = 0

    return action


def format_observation(obs_dict: dict) -> str:
    """Format an observation dict into a concise string for the LLM."""
    inv = obs_dict["current_inventory"]
    dem = obs_dict["current_demand"]

    pending = obs_dict.get("pending_shipments", [])
    if pending:
        pending_lines = []
        for s in pending:
            pending_lines.append(
                f"  {s['origin']}->{s['destination']}: "
                f"{s['amount']}u, {s['steps_remaining']} steps ({s['speed_mode']})"
            )
        pending_str = "\n".join(pending_lines)
    else:
        pending_str = "  None"

    return (
        f"Step {obs_dict['step_number']}/{obs_dict['total_steps']}\n"
        f"Inventory: Seattle={inv['Seattle']:.1f}  Chicago={inv['Chicago']:.1f}  NYC={inv['NYC']:.1f}\n"
        f"Demand:    Seattle={dem['Seattle']:.1f}  Chicago={dem['Chicago']:.1f}  NYC={dem['NYC']:.1f}\n"
        f"Pending shipments:\n{pending_str}\n"
        f"Carbon remaining: {obs_dict['carbon_credit_balance']:.1f}\n"
        f"Cumulative profit: {obs_dict['cumulative_profit']:.2f}\n"
        f"Cumulative carbon: {obs_dict['cumulative_carbon']:.2f}\n"
        f"Weather: {obs_dict.get('weather_alert') or 'Clear'}\n\n"
        f"Output your JSON action:"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Run a single task episode
# ═══════════════════════════════════════════════════════════════════════════

def run_episode(client: OpenAI, task_id: str) -> dict:
    """
    Run one full episode: reset env, loop step-by-step with LLM, grade.
    Returns dict with task_id, score, feedback, metrics, total_reward.
    """
    task_def = TASKS[task_id]

    print(f"\n{'='*60}")
    print(f"TASK: {task_id} ({task_def.difficulty.value})")
    print(f"  {task_def.description}")
    print(f"{'='*60}")

    # Initialize environment directly (not over HTTP)
    env = EcoLogisticsEnv(seed=SEED)
    obs = env.reset(task_id=task_id, seed=SEED)

    obs_dict = obs.model_dump()
    print(f"  Initial inventory: {obs_dict['current_inventory']}")
    print(f"  Carbon budget:     {obs_dict['carbon_credit_balance']}")
    print(f"  Total steps:       {obs_dict['total_steps']}")

    # Build system prompt with task-specific description
    system_prompt = SYSTEM_PROMPT.format(task_description=task_def.description)
    messages = [{"role": "system", "content": system_prompt}]

    done = False
    step_num = 0
    total_reward = 0.0

    while not done:
        # Build user message from current observation
        obs_dict = obs.model_dump()
        user_msg = format_observation(obs_dict)
        messages.append({"role": "user", "content": user_msg})

        # Call LLM via OpenAI client
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            raw = response.choices[0].message.content or ""
            action_dict = parse_action(raw)
        except Exception as e:
            print(f"  [Step {step_num}] LLM error: {e}. Using fallback.")
            action_dict = FALLBACK_ACTION.copy()

        messages.append({"role": "assistant", "content": json.dumps(action_dict)})

        # Convert to pydantic Action and step the environment
        action = Action(
            ship_amount=action_dict["ship_amount"],
            origin_city=action_dict["origin_city"],
            destination_city=action_dict["destination_city"],
            speed_mode=SpeedMode(action_dict["speed_mode"]),
        )

        obs, reward, done, info = env.step(action)
        total_reward += reward.total

        # Log
        if DEBUG:
            ship_info = (
                f"{action_dict['ship_amount']:.0f}u "
                f"{action_dict['origin_city']}->{action_dict['destination_city']} "
                f"({action_dict['speed_mode']})"
                if action_dict["ship_amount"] > 0
                else "no-op"
            )
            inv = obs.current_inventory
            print(
                f"  Step {step_num:2d}: {ship_info:40s} | "
                f"reward={reward.total:+7.2f} | "
                f"inv=[{inv['Seattle']:.0f}, {inv['Chicago']:.0f}, {inv['NYC']:.0f}]"
            )

        step_num += 1

        # Truncate message history to stay within context limits
        if len(messages) > MAX_HISTORY:
            messages = messages[:1] + messages[-(MAX_HISTORY - 1):]

    # Grade the episode
    grade = env.grade()

    print(f"\n  RESULT: score={grade.score:.4f}")
    print(f"  Feedback: {grade.feedback}")
    print(f"  Total reward: {total_reward:.2f}")

    return {
        "task_id": task_id,
        "score": grade.score,
        "feedback": grade.feedback,
        "metrics": grade.metrics,
        "total_reward": round(total_reward, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Eco-Logistics — LLM Inference Script")
    print("=" * 60)

    # Validate required env vars
    if not API_KEY:
        print("ERROR: HF_TOKEN (or API_KEY) environment variable is not set.")
        print("  export HF_TOKEN='your-api-key-here'")
        sys.exit(1)

    if not MODEL_NAME:
        print("ERROR: MODEL_NAME environment variable is not set.")
        print("  export MODEL_NAME='gpt-4o'")
        sys.exit(1)

    print(f"  API_BASE_URL : {API_BASE_URL}")
    print(f"  MODEL_NAME   : {MODEL_NAME}")
    print(f"  HF_TOKEN     : {'***' + API_KEY[-4:] if len(API_KEY) > 4 else '(set)'}")

    # Initialize OpenAI client
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    # Run all tasks
    results = []
    start_time = time.time()

    for task_id in TASK_IDS:
        result = run_episode(client, task_id)
        results.append(result)

    elapsed = time.time() - start_time

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "PASS" if r["score"] > 0 else "FAIL"
        print(
            f"  [{status}] {r['task_id']:25s}  "
            f"score={r['score']:.4f}  "
            f"reward={r['total_reward']:+.2f}"
        )

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0
    print(f"\n  Average score : {avg_score:.4f}")
    print(f"  Total time    : {elapsed:.1f}s")
    print(f"  All in [0,1]  : {all(0.0 <= r['score'] <= 1.0 for r in results)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
