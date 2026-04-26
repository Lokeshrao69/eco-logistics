"""
Eco-Logistics v2 — Inference helpers.

Role-aware prompt construction + retry loop for emitting valid 10/25-step
JSON plans. Used both by the v2 demo Space and by the v2 GRPO rollout loop.

Key functions:
- build_role_prompt(role, obs)         → returns ChatML string for a role
- parse_plan_or_fallback(text, target) → returns (List[PlanStep], was_valid)
- generate_plan_with_retry(...)        → retry 3x with temp 0.7→0.3→0.1
- heuristic_role_plan(role, obs, n)    → deterministic fallback per role

Backward compat: the v1-style single-agent prompt builders also live here
under the same names the v8 notebook uses (build_chat_prompt,
format_observation_prompt, parse_action_array) so v8 can keep working.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from models import (
    CITIES,
    Action,
    AgentPlan,
    AgentRole,
    MultiAgentObservation,
    Observation,
    PlanStep,
    SpeedMode,
)

# ─────────────────────────────────────────────────────────────────────────────
# v2 SYSTEM PROMPT  (role-aware)
# ─────────────────────────────────────────────────────────────────────────────

V2_SYSTEM_PROMPT = (
    "You are a supply chain AI managing 3 warehouses. "
    "Output ONLY valid JSON array of 10-step plans. Each step: "
    "ship_amount, origin_city, destination_city, speed_mode ('Rail' or 'Air'). "
    "If REVISE flag, output revised remaining steps. "
    "Roles: seattle_mgr, chicago_router, nyc_carbon. "
    "You can propose to another role: "
    "{'proposal_to': 'chicago_router', 'offer': {...}}. "
    "Always prefer rail for long-haul. Parse weather_alert and plan defensively."
)

# Role-specific guidance appended to system prompt
ROLE_GUIDANCE: Dict[str, str] = {
    AgentRole.SEATTLE_MGR.value: (
        "Your specialty: Seattle inventory management and outbound shipping. "
        "Prioritize keeping Seattle stock between 30-60 units. "
        "When Chicago/NYC are low, ship FROM Seattle (you are the source). "
        "Use rail unless demand spike requires speed."
    ),
    AgentRole.CHICAGO_ROUTER.value: (
        "Your specialty: Chicago hub routing — you are the central waypoint. "
        "Balance flow Seattle→Chicago→NYC. Re-route when weather hits one path. "
        "If Seattle and NYC need balance, you can bridge stock through Chicago."
    ),
    AgentRole.NYC_CARBON.value: (
        "Your specialty: NYC inventory + GLOBAL carbon optimization. "
        "Veto high-carbon shipments when total carbon is rising. "
        "Prefer rail aggressively. Coordinate with seattle_mgr to bulk-ship via rail "
        "(you can propose: {'proposal_to': 'seattle_mgr', 'offer': {'route': 'Seattle→Chicago', 'mode': 'Rail'}})."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# v1 BACKWARD-COMPAT (used by v8 notebook + main.py)
# ─────────────────────────────────────────────────────────────────────────────

V1_SYSTEM_PROMPT = (
    "You are a supply chain AI managing 3 warehouses (Seattle, Chicago, NYC). "
    "You will receive an initial observation. Output ONLY a valid JSON array of "
    "10 actions for the full episode. Each action MUST have these exact fields: "
    "ship_amount (number), origin_city (string), destination_city (string), "
    "speed_mode (string, either 'Rail' or 'Air'). "
    "Always prefer Rail for long-haul shipments to reduce carbon. "
    "Parse weather_alert text and route defensively around disrupted lanes."
)


def format_observation_prompt(obs: Dict[str, Any] | Observation) -> str:
    """Render an observation dict into a compact text body for the user turn.

    v1 helper, kept for backward compat with v8 notebook.
    """
    if hasattr(obs, "model_dump"):
        obs = obs.model_dump()
    return (
        f"Initial state:\n"
        f"  Inventory: {obs.get('current_inventory', {})}\n"
        f"  Demand:    {obs.get('current_demand', {})}\n"
        f"  Pending:   {obs.get('pending_shipments', [])}\n"
        f"  Carbon balance: {obs.get('carbon_credit_balance', 0)}\n"
        f"  Step: {obs.get('step_number', 0)} / {obs.get('total_steps', 10)}\n"
        f"  Weather: {obs.get('weather_alert') or 'none'}\n"
        f"\nOutput a JSON array of 10 actions for the full episode."
    )


def build_chat_prompt(user_body: str, system: str = V1_SYSTEM_PROMPT) -> str:
    """v1 ChatML prompt builder. Used by v8 notebook."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_body}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def parse_action_array(completion: str) -> Tuple[List[Dict[str, Any]], bool]:
    """v1 parser: extract list of 10 action dicts from completion text.

    Returns (parsed_list, was_valid). On failure, returns 10 no-op actions.
    """
    SAFE_FALLBACK = [
        {"ship_amount": 0.0, "origin_city": "Seattle", "destination_city": "Chicago", "speed_mode": "Rail"}
        for _ in range(10)
    ]
    if not completion:
        return SAFE_FALLBACK, False

    match = re.search(r"\[.*\]", completion, re.DOTALL)
    if not match:
        return SAFE_FALLBACK, False

    try:
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, list):
            return SAFE_FALLBACK, False
        cleaned: List[Dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                cleaned.append({
                    "ship_amount": float(item.get("ship_amount", 0.0)),
                    "origin_city": str(item.get("origin_city", "Seattle")),
                    "destination_city": str(item.get("destination_city", "Chicago")),
                    "speed_mode": str(item.get("speed_mode", "Rail")),
                })
            except Exception:
                continue
        if len(cleaned) == 0:
            return SAFE_FALLBACK, False
        # Pad/truncate to 10
        while len(cleaned) < 10:
            cleaned.append(SAFE_FALLBACK[0])
        return cleaned[:10], True
    except (json.JSONDecodeError, ValueError):
        return SAFE_FALLBACK, False


# ─────────────────────────────────────────────────────────────────────────────
# v2 ROLE PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def format_multiagent_obs_prompt(mao: MultiAgentObservation) -> str:
    """Render a MultiAgentObservation into a compact prompt body.

    Includes role context, peer recent actions, replanning flag, curriculum
    level, and any open negotiation proposals.
    """
    base = mao.base
    obs_dict = base.model_dump() if hasattr(base, "model_dump") else dict(base)

    revise_note = ""
    if mao.revise_plan_flag:
        revise_note = (
            "\n[REVISE PLAN flag is SET. Output a revised plan covering ONLY the "
            "remaining steps from the current step to the end of the episode.]"
        )

    peer_lines = ""
    if mao.other_agents_recent_actions:
        peer_lines = "\n  Peers' recent actions:"
        for entry in mao.other_agents_recent_actions[-5:]:
            a = entry.get("action", {})
            peer_lines += (
                f"\n    - {entry.get('role')} @ step {entry.get('step')}: "
                f"ship {a.get('ship_amount', 0)} {a.get('origin_city', '?')}"
                f"→{a.get('destination_city', '?')} via {a.get('speed_mode', '?')}"
            )

    negotiation_line = ""
    if mao.last_negotiation_status.value != "none":
        negotiation_line = f"\n  Last negotiation status: {mao.last_negotiation_status.value}"

    return (
        f"Role: {mao.role.value}\n"
        f"Curriculum level: {mao.curriculum_level} "
        f"(recent success rate: {mao.curriculum_recent_success_rate:.2f})\n"
        f"Initial state:\n"
        f"  Inventory: {obs_dict.get('current_inventory', {})}\n"
        f"  Demand:    {obs_dict.get('current_demand', {})}\n"
        f"  Pending:   {obs_dict.get('pending_shipments', [])}\n"
        f"  Carbon balance: {obs_dict.get('carbon_credit_balance', 0)}\n"
        f"  Step: {obs_dict.get('step_number', 0)} / {obs_dict.get('total_steps', 25)}\n"
        f"  Weather: {obs_dict.get('weather_alert') or 'none'}"
        f"{peer_lines}{negotiation_line}{revise_note}\n"
        f"\nOutput a JSON array of plan steps for the remaining episode. "
        f"You may also include a 'negotiation_proposal' field at the END of "
        f"the array as a separate object: "
        f"{{'proposal_to': '<other_role>', 'offer': {{'route': 'X→Y', 'mode': 'Rail'}}}}."
    )


def build_role_prompt(
    role: AgentRole,
    mao: MultiAgentObservation,
) -> str:
    """Build a ChatML prompt for a specific role."""
    role_str = role.value
    guidance = ROLE_GUIDANCE.get(role_str, "")
    full_system = V2_SYSTEM_PROMPT + "\n\n" + guidance

    user_body = format_multiagent_obs_prompt(mao)
    return (
        f"<|im_start|>system\n{full_system}<|im_end|>\n"
        f"<|im_start|>user\n{user_body}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# v2 PLAN PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_plan_or_fallback(
    completion: str,
    target_length: int = 10,
    role: AgentRole = AgentRole.SOLO,
) -> Tuple[AgentPlan, Dict[str, Any], bool]:
    """Parse model completion into an AgentPlan.

    Returns (plan, negotiation_proposal_or_None, was_valid).
    On failure, returns a no-op plan + no proposal + was_valid=False.
    """
    fallback_plan = AgentPlan(
        role=role,
        steps=[
            PlanStep(
                ship_amount=0.0,
                origin_city="Seattle",
                destination_city="Chicago",
                speed_mode=SpeedMode.RAIL,
            )
            for _ in range(target_length)
        ],
        is_revision=False,
        starting_step=0,
    )

    if not completion:
        return fallback_plan, None, False

    # Try to find a top-level JSON array
    match = re.search(r"\[.*\]", completion, re.DOTALL)
    if not match:
        return fallback_plan, None, False

    try:
        parsed = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return fallback_plan, None, False

    if not isinstance(parsed, list) or len(parsed) == 0:
        return fallback_plan, None, False

    # Last element MAY be a negotiation proposal — detect by presence of 'proposal_to' key
    negotiation_proposal: Optional[Dict[str, Any]] = None
    if isinstance(parsed[-1], dict) and "proposal_to" in parsed[-1]:
        negotiation_proposal = parsed[-1]
        parsed = parsed[:-1]

    cleaned_steps: List[PlanStep] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            origin = str(item.get("origin_city", "Seattle"))
            dest = str(item.get("destination_city", "Chicago"))
            mode_str = str(item.get("speed_mode", "Rail"))
            # Coerce speed mode
            if mode_str.lower() in ("rail", "train"):
                mode = SpeedMode.RAIL
            elif mode_str.lower() in ("air", "plane"):
                mode = SpeedMode.AIR
            else:
                mode = SpeedMode.RAIL
            # Validate cities
            if origin not in CITIES:
                origin = "Seattle"
            if dest not in CITIES:
                dest = "Chicago"
            cleaned_steps.append(PlanStep(
                ship_amount=max(0.0, float(item.get("ship_amount", 0.0))),
                origin_city=origin,
                destination_city=dest,
                speed_mode=mode,
            ))
        except Exception:
            continue

    if len(cleaned_steps) == 0:
        return fallback_plan, None, False

    # Pad / truncate to target_length
    while len(cleaned_steps) < target_length:
        cleaned_steps.append(PlanStep(
            ship_amount=0.0, origin_city="Seattle", destination_city="Chicago",
            speed_mode=SpeedMode.RAIL,
        ))
    cleaned_steps = cleaned_steps[:target_length]

    plan = AgentPlan(
        role=role,
        steps=cleaned_steps,
        is_revision=False,
        starting_step=0,
        negotiation_proposal=negotiation_proposal,
    )
    return plan, negotiation_proposal, True


# ─────────────────────────────────────────────────────────────────────────────
# v2 RETRY LOOP
# ─────────────────────────────────────────────────────────────────────────────

def generate_plan_with_retry(
    generate_fn: Callable[[str, float], str],
    prompt: str,
    role: AgentRole = AgentRole.SOLO,
    target_length: int = 10,
    max_attempts: int = 3,
    temps: Optional[List[float]] = None,
) -> Tuple[AgentPlan, bool, int]:
    """Call generate_fn(prompt, temperature) up to max_attempts times.

    Temperature schedule: 0.7 → 0.3 → 0.1 (less random each retry).
    Returns first valid AgentPlan, or falls back to heuristic if all fail.

    Returns: (plan, was_model_valid, attempts_used).
    """
    if temps is None:
        temps = [0.7, 0.3, 0.1]
    while len(temps) < max_attempts:
        temps.append(0.1)

    for attempt in range(max_attempts):
        temp = temps[attempt]
        try:
            completion = generate_fn(prompt, temp)
        except Exception:
            completion = ""
        plan, _, valid = parse_plan_or_fallback(completion, target_length, role)
        if valid:
            return plan, True, attempt + 1

    # All attempts failed — use heuristic fallback
    return heuristic_role_plan_no_obs(role, target_length), False, max_attempts


def heuristic_role_plan_no_obs(
    role: AgentRole,
    target_length: int = 10,
) -> AgentPlan:
    """Role-based deterministic plan when no observation is available.

    Each role has its own default routing pattern:
    - seattle_mgr: ships Seattle → Chicago every step
    - chicago_router: ships Chicago → NYC every step
    - nyc_carbon: no-op (carbon-conservative default)
    """
    role_str = role.value
    if role_str == AgentRole.SEATTLE_MGR.value:
        steps = [PlanStep(
            ship_amount=2.0, origin_city="Seattle", destination_city="Chicago",
            speed_mode=SpeedMode.RAIL,
        ) for _ in range(target_length)]
    elif role_str == AgentRole.CHICAGO_ROUTER.value:
        steps = [PlanStep(
            ship_amount=1.5, origin_city="Chicago", destination_city="NYC",
            speed_mode=SpeedMode.RAIL,
        ) for _ in range(target_length)]
    elif role_str == AgentRole.NYC_CARBON.value:
        steps = [PlanStep(
            ship_amount=0.0, origin_city="NYC", destination_city="Seattle",
            speed_mode=SpeedMode.RAIL,
        ) for _ in range(target_length)]
    else:
        # SOLO fallback: rebalance from Seattle to Chicago
        steps = [PlanStep(
            ship_amount=1.0, origin_city="Seattle", destination_city="Chicago",
            speed_mode=SpeedMode.RAIL,
        ) for _ in range(target_length)]

    return AgentPlan(
        role=role,
        steps=steps,
        is_revision=False,
        starting_step=0,
    )


def heuristic_role_plan(
    role: AgentRole,
    obs: Observation | MultiAgentObservation,
    target_length: int = 10,
) -> AgentPlan:
    """Observation-aware heuristic plan per role.

    Smarter than the no-obs version: uses current inventory + demand to decide
    ship_amount and route per role.
    """
    if isinstance(obs, MultiAgentObservation):
        base = obs.base
    else:
        base = obs

    inv = base.current_inventory
    demand = base.current_demand
    role_str = role.value

    steps: List[PlanStep] = []

    if role_str == AgentRole.SEATTLE_MGR.value:
        # Ship Seattle → wherever has lowest stock relative to demand
        for _ in range(target_length):
            target_city = min(["Chicago", "NYC"], key=lambda c: inv.get(c, 0) - demand.get(c, 0))
            available = max(0.0, inv.get("Seattle", 0) - 25)  # keep Seattle min stock
            ship = min(5.0, available * 0.3)
            steps.append(PlanStep(
                ship_amount=round(ship, 1),
                origin_city="Seattle",
                destination_city=target_city,
                speed_mode=SpeedMode.RAIL,
            ))

    elif role_str == AgentRole.CHICAGO_ROUTER.value:
        # Ship Chicago → NYC primarily
        for _ in range(target_length):
            available = max(0.0, inv.get("Chicago", 0) - 25)
            ship = min(3.0, available * 0.3)
            steps.append(PlanStep(
                ship_amount=round(ship, 1),
                origin_city="Chicago",
                destination_city="NYC",
                speed_mode=SpeedMode.RAIL,
            ))

    elif role_str == AgentRole.NYC_CARBON.value:
        # Carbon-conservative: only ship if NYC has huge surplus, otherwise no-op
        nyc_stock = inv.get("NYC", 0)
        for _ in range(target_length):
            if nyc_stock > 60:
                steps.append(PlanStep(
                    ship_amount=2.0,
                    origin_city="NYC",
                    destination_city="Chicago",
                    speed_mode=SpeedMode.RAIL,
                ))
            else:
                steps.append(PlanStep(
                    ship_amount=0.0,
                    origin_city="NYC",
                    destination_city="Seattle",
                    speed_mode=SpeedMode.RAIL,
                ))

    else:
        # SOLO: simple rebalance from richest to poorest city
        cities_sorted = sorted(CITIES, key=lambda c: inv.get(c, 0))
        max_c = cities_sorted[-1]
        min_c = cities_sorted[0]
        for _ in range(target_length):
            gap = inv.get(max_c, 0) - inv.get(min_c, 0)
            ship = min(gap * 0.4, inv.get(max_c, 0) * 0.3) if gap > 15 else 0.0
            steps.append(PlanStep(
                ship_amount=round(max(0.0, ship), 1),
                origin_city=max_c,
                destination_city=min_c,
                speed_mode=SpeedMode.RAIL,
            ))

    return AgentPlan(
        role=role,
        steps=steps,
        is_revision=False,
        starting_step=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# v2 MULTI-AGENT ORCHESTRATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_role_plans(
    generate_fn: Callable[[str, float], str],
    obs_per_role: Dict[str, MultiAgentObservation],
    target_length: int = 10,
    max_attempts: int = 3,
) -> Tuple[Dict[str, AgentPlan], Dict[str, bool], Dict[str, int]]:
    """Generate plans for all 3 roles, one role at a time.

    Returns:
      plans: {role_str: AgentPlan}
      valid_per_role: {role_str: True if model emitted valid plan}
      attempts_per_role: {role_str: how many attempts used}
    """
    plans: Dict[str, AgentPlan] = {}
    valid_map: Dict[str, bool] = {}
    attempts_map: Dict[str, int] = {}

    for role_str, mao in obs_per_role.items():
        try:
            role_enum = AgentRole(role_str)
        except ValueError:
            continue

        prompt = build_role_prompt(role_enum, mao)
        plan, was_valid, attempts = generate_plan_with_retry(
            generate_fn=generate_fn,
            prompt=prompt,
            role=role_enum,
            target_length=target_length,
            max_attempts=max_attempts,
        )
        # If model failed all retries, replace with observation-aware heuristic
        if not was_valid:
            plan = heuristic_role_plan(role_enum, mao, target_length)

        plans[role_str] = plan
        valid_map[role_str] = was_valid
        attempts_map[role_str] = attempts

    return plans, valid_map, attempts_map