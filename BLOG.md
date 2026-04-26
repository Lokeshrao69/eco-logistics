# From v8 to v9: Engineering an LLM into a Supply Chain Planner

**Eco-Logistics — OpenEnv Hackathon Round 2 (Crystal Blue)**

This is the writeup behind our submission to the OpenEnv Hackathon's "World Modeling for Professional Tasks" track. We trained Qwen-2.5-1.5B with GRPO to plan 25-step trajectories in a 3-warehouse supply-chain environment under a strict carbon budget. v9 beats our prior v8 submission across all three task graders, hits 100% format compliance, and ships at https://huggingface.co/lokeshrao226/eco-logistics-qwen-grpo-v2.

But the real value isn't the headline numbers — it's the process. We failed three times before v9 worked, and each failure taught us something we'd have missed otherwise. Here's the honest version.

## TL;DR

| | v8 (prior) | v9 (this) | Change |
|---|:-:|:-:|:-:|
| Restock Only grader | 0.13 | **0.999** | +669% |
| Inventory Balanced grader | 0.19 | **0.269** | +42% |
| Net-Zero Profit grader | 0.273 | **0.338** | +24% |
| Valid action rate | 20% | **100%** | 5× |

The technical changes in v9: 25-step task variants, curriculum learning, receding-horizon replanning, multi-agent endpoints, **constrained decoding**. The policy that ships is the LoRA + parser together — a key design choice we explain below.

---

## Section 1: What we built

Eco-Logistics is a 3-warehouse simulator (Seattle, Chicago, NYC). An agent receives observations (inventory, demand, pending shipments, carbon budget, weather alerts) and emits actions (ship X units from origin to destination via Rail or Air). Three tasks of escalating difficulty:

- **Restock Only**: keep all warehouses ≥ 20 units. Easy. Stable demand.
- **Inventory Balanced**: keep stock spread within 10% across cities. Medium. Seasonal demand.
- **Net-Zero Profit**: maximize profit under a strict carbon cap. Hard. Volatile demand + weather shocks.

The hard task is where the interesting policies live. With Air shipments costing 6–12 carbon per unit and a budget of 100, you can't ship freely. The agent has to plan around weather, demand spikes, and inventory decay over a 25-step horizon while staying inside the budget.

Our v8 submission trained an LLM to do this with GRPO. Got 0.273 grader. Submitted, called it done. Then we kept going.

---

## Section 2: Three failures before v9 worked

### Failure 1: Curriculum overfitting (v9 round 1)

We thought: "the env has 3 tasks of escalating difficulty, let's let curriculum learning handle the order. Start at easy, advance when 80% pass rate."

Result: held-out eval grader 0.078 on net_zero_profit. Worse than v8.

What happened: the curriculum spent ~80% of training time on `restock_only` (which the model nailed at 0.999) and only briefly touched `net_zero_profit`. The model became a **specialist at the easiest task** and never learned the carbon tradeoff. When we evaluated on the hard task held-out, it was clueless.

Lesson: **curriculum learning works when the harder tasks share structure with the easier ones**. They don't, here. `restock_only` is "ship enough to stay above threshold." `net_zero_profit` is "ship the minimum needed to extract profit under a budget cliff." Different objectives.

### Failure 2: Dense rewards collapsing (the SFT-then-GRPO experiment)

Before v9, we tried bootstrapping with SFT (supervised fine-tuning on heuristic trajectories), then GRPO on top. Logic: heuristic gives valid actions, SFT teaches the format, GRPO refines the policy.

What we got:
- After SFT: 80% format compliance (great!), 0.226 grader
- After GRPO with multi-component reward (profit + delivery + carbon shaping): grader **collapsed to 0.146**
- After GRPO with grader-only reward: still collapsed

Lesson: **GRPO needs variance among samples**. SFT made the policy too uniform — all 4 generations per prompt looked similar, GRPO had no signal to refine. The "smoother" multi-component rewards just told the model to ship more (more profit), which broke carbon. Single grader-only reward turned out to be the only stable config.

We documented this as our negative result in the v8 writeup.

### Failure 3: 422 errors and over-shipping (v9 round 2)

After abandoning SFT, we trained pure GRPO on v2 (25-step tasks, force-hard-task probability 0.5). Training looked great:

```
step 5:  reward=90    grader=0.118
step 10: reward=319   grader=0.164
step 15: reward=1975  grader=0.495   <- training peak
```

Held-out eval: **net_zero_profit grader 0.144**. Half what training suggested.

We dug into per-seed plans. Found two issues:

**Issue A**: 50% of GRPO rollouts hit `422 Unprocessable Entity` when submitting plans. The model was emitting invalid city names ("Boston", "Paris") that our pydantic schema rejected. Our parser was clamping `ship_amount` and coercing `speed_mode`, but **didn't validate cities**. Half the gradient signal was being wasted on rejected plans.

**Issue B**: When plans did get through, the model preferred Air shipments. We logged the math:
- 5 units × 25 steps × Air-Seattle→NYC carbon (12) = **1500 carbon used**
- Budget: 100
- Result: grader 0.001 (penalty floor)

The model wasn't *deciding* poorly. It was **physically incapable** of ranging a valid policy under our parser, because the parser let it overshoot.

### Fix: constrained decoding

Sat down and did the carbon arithmetic. With carbon budget 100 over 25 steps via cheapest rail (Seattle→Chicago, 2.0 carbon/unit):

```
budget / (steps × carbon_per_unit) = 100 / (25 × 2.0) = 2.0 ship_units/step
```

So the **maximum sustainable ship_amount is ~2 per step**. Our parser cap was 5. That was the bug.

We made the parser:
1. Coerce invalid cities to canonical ones via substring match
2. Force `speed_mode = "Rail"` always
3. Cap `ship_amount` at 2.0

```python
def parse_plan(completion, target_length=25):
    # ... parse JSON, coerce city names ...
    cleaned.append({
        "ship_amount": min(2.0, max(0.0, float(item.get("ship_amount", 0.0)))),
        "origin_city": _coerce_city(item.get("origin_city", ""), "Seattle"),
        "destination_city": _coerce_city(item.get("destination_city", ""), "Chicago"),
        "speed_mode": "Rail",   # FORCED — Air carbon kills the budget
    })
```

This converts an unconstrained generative LLM into a **policy that emits feasibility-respecting actions**. The LoRA still chooses what city to ship from, where, and how much (within 0–2). The constraint just removes the carbon-blowing degrees of freedom.

**Result with the constrained parser** (greedy eval, 30 held-out episodes):

| Task | Grader |
|---|:-:|
| Restock Only | 0.999 |
| Inventory Balanced | 0.269 |
| Net-Zero Profit | **0.338** |
| Valid action rate | **100%** |

That's our v9.

---

## Section 3: Is constrained decoding "real" learning?

A reasonable concern: by hard-coding Rail and capping ship≤2, are we just bypassing the LLM and shipping a parser?

We thought about this carefully. Three things to note:

1. **The LoRA matters.** Replace it with the base Qwen + same parser, and net_zero_profit grader drops to ~0.18. The trained policy is doing real work — choosing routes, ship amounts, and timing — within the constraint set.

2. **This is how RL-trained LLMs are deployed in production.** LangChain agents, ToolLLM, function-calling models all wrap raw LLM output in a validator/projector layer. We're not doing anything they aren't.

3. **The constraints are derived from the environment, not made up.** `ship ≤ 2` comes from the carbon arithmetic. It's not a flattering hyperparameter — it's the feasible boundary.

That said, we acknowledge this in our model card: "the LoRA + parser are a single policy."

---

## Section 4: The honest weaknesses

**The no-op exploit.** A "ship nothing" policy on net_zero_profit posts grader 0.465 — *higher* than v8 (0.273), heuristic (0.292), and v9 (0.338). Why? The grader is `profit / max_expected_profit`, with a quadratic penalty for over-budget carbon. A no-op uses zero carbon → no penalty. Demand still gets partially fulfilled from initial inventory + restock. Net result: the model shows up to a "task" and wins by not playing.

We didn't fix the grader because v8 was scored against the same grader and we want apples-to-apples comparison. But we'd flag this for next iteration: a profit floor (require ≥50% of max profit before accepting any score above 0.3) would close it.

Note: **our v9 (0.338) and the heuristic (0.292) both beat v8 (0.273) on this grader.** All three are below the no-op exploit at 0.465 only because the grader has this specific geometry. We're not gaming it; we're transparent about it.

**Training is volatile.** Our v9 run hit two transient collapses (grader → 0.001, valid → 0%) at steps 45 and 55 of the per-batch reward call counter. The model recovered both times. We didn't stop early — we trusted held-out eval as ground truth, and at step 40 it passed.

**Multi-agent endpoints unused in submission.** The v2 architecture supports 3-role coalitions (`seattle_mgr`, `chicago_router`, `nyc_carbon`) with a negotiation protocol. We tested those endpoints work end-to-end, but trained the LoRA single-agent because we ran out of GPU budget. Multi-agent training is on the future-work list.

---

## Section 5: What's next

Three things we'd build with more time:

1. **Replanning during training.** v9 trained the model to emit a single 25-step plan upfront. v2 supports `submit_revised_plan` for mid-episode replanning, but we never trained it. Adding mid-rollout revisions to the GRPO loop would teach the model to react to weather shocks dynamically.

2. **Multi-agent training.** The 3-role coalition endpoints are deployed. A multi-agent GRPO run with role-specific reward shaping would test whether the negotiation bonuses (`+8` for accepted bilateral proposals) transfer to learned coordination.

3. **Soft constraints instead of hard.** Constrained decoding works but is brittle — change the carbon budget and our hard cap of 2 breaks. A learned "carbon-respecting head" or a dual-head model that emits both a plan and a feasibility check would generalize better.

---

## Section 6: Reproducing

Everything is open. The trained LoRA, training notebook, constrained parser code, and held-out eval results are all in:

- LoRA: https://huggingface.co/lokeshrao226/eco-logistics-qwen-grpo-v2
- Code: https://github.com/Lokeshrao69/eco-logistics
- Live env: https://huggingface.co/spaces/lokeshrao226/eco-logistics
- Training notebook: `train_eco_logistics_grpo_v9_FINAL_v2.ipynb` in the repo

T4 GPU is enough. Training takes ~60 min. Held-out eval takes ~20 min for 30 episodes.

---

## Closing

The lesson we keep learning: **engineering wins in LLM-RL come from understanding the environment, not from bigger models or fancier algorithms**. Our v9 isn't using a different model than v8 (still Qwen-2.5-1.5B) or a different algorithm (still GRPO). What changed is that we did the carbon arithmetic, found the failure modes empirically, and fixed them with a constrained parser. That's it.

We're shipping v9.

— Crystal Blue (J Lokesh Rao)
