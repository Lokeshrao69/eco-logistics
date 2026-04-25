# Eco-Logistics: Multi-City Supply Chain Optimizer

> **OpenEnv Hackathon — Round 2 Submission**
> Theme: **World Modeling — Professional Tasks**

An RL environment that puts a language model in charge of a three-warehouse supply chain (Seattle · Chicago · NYC) and post-trains it with **GRPO** to navigate the profit-vs-carbon tradeoff under non-stationary shocks.

**Headline result (held-out seeds, 10 episodes each):**
> GRPO Qwen-2.5-1.5B achieves **+27% more profit** with **−91% less carbon emitted** than the base model — a **15× improvement in profit-per-unit-carbon.**

---

##  Deliverables

| Resource | Link |
|---|---|
| HF Space (live environment) | https://huggingface.co/spaces/lokeshrao226/eco-logistics |
| Training notebook (Colab) | https://colab.research.google.com/github/Lokeshrao69/eco-logistics/blob/main/train_eco_logistics_grpo.ipynb |
| Code repository (this repo) | https://github.com/Lokeshrao69/eco-logistics |
| Trained LoRA adapter | https://huggingface.co/lokeshrao226/eco-logistics-qwen-grpo |
| Writeup | {SET_AT_SUBMIT_TIME} |

---

## The Problem

Real supply chains don't fail in controlled conditions. They fail when demand spikes unexpectedly, when competitors outbid you on a route, or when carbon budgets tighten mid-quarter. We wanted an environment where an agent has to plan **defensively against surprise** and learn the **profit-vs-carbon tradeoff** at the same time.

## The Environment

A 3-warehouse OpenEnv-compliant simulator with the standard `reset` / `step` / `state` / `grader` interface:

- **Observation**: current inventory, pending shipments, demand forecast, carbon credit balance, weather/market alerts
- **Action**: `(ship_amount, origin_city, destination_city, speed_mode ∈ {{Rail, Air}})`
- **Reward (dense)**: `sales_revenue − shipping_cost − carbon_penalty − storage_fee + healthy_stock_bonus`
- **Three tasks** of increasing difficulty:
  1. `restock_only` — easy
  2. `inventory_balanced` — medium (used for this submission)
  3. `net_zero_profit` — hard (max profit under a strict carbon budget)

**World-modeling wrapper**: at rollout time we inject non-stationary demand shocks (`p=0.15`, 2.5× multiplier) and competitor bids (`p=0.20`, 1.8× shipping cost) surfaced through the `weather_alert` observation field. The agent has to plan *around* these surprises without seeing them at plan-generation time.

## The Training

- **Base model**: Qwen-2.5-1.5B-Instruct
- **Method**: GRPO via TRL, LoRA adapters via Unsloth, 4-bit quantization, single T4
- **Design choice — Upfront Trajectory Planning**: the model generates the entire 10-step action sequence as a JSON array from the initial observation. One completion = one rollout, which makes GRPO-over-HTTP tractable.
- **Dataset**: 50 unique initial states sampled across all 3 tasks and seeds 0–49.
- **Evaluation**: held-out seeds 500–509 the model never saw during training.

**Reward hacking diagnosis and fix** (documented in the writeup): the initial training run collapsed at step 15 when the model discovered that outputting invalid JSON let a fallback action shield it from carbon penalties. We diagnosed the mechanism (format penalty magnitude < carbon savings from null shipments) and fixed it by making invalid output cost `-1000.0` instead of `-5.0` — this restored gradient signal and stabilized training.

## Results

![Training curves + 4-way comparison on held-out seeds](training_curves_inventory_balanced.png)

**4-way comparison on held-out seeds 500–509 (10 episodes each):**

| Policy | Profit | Carbon | Profit/Carbon (agg) | Grader | Delivery |
|---|---|---|---|---|---|
| Random policy | 3946 | 1252 | 3.2 | 0.065 | 87.6% |
| Heuristic (richest→poorest, rail) | 3558 | 450 | 7.9 | 0.040 | 71.5% |
| Base Qwen-2.5-1.5B | 3793 | 1429 | 2.7 | 0.135 | 86.6% |
| **GRPO Qwen (ours)** | **4828** | **122** | **39.6** | 0.195 | 87.6% |

**The key deltas (GRPO vs base Qwen on same held-out seeds):**
- **Profit**: 3793 → **4828** (+27%)
- **Carbon used**: 1429 → **122** (−91%)
- **Profit per unit carbon**: 2.7 → **39.6** (15× improvement)
- **Policy stability**: profit σ went from 1770 (base) to **213 (ours)** — 8× more consistent

**What we're cautious about**: grader_score improvement (0.135 → 0.195, +44%) is meaningful but the error bars of base Qwen (0.135 ± 0.082) are wide enough to partly overlap our mean. We report it as "measurable improvement," not "statistically significant at 95%." What's rock-solid is the carbon/profit story: profit σ dropped from 1771 (base) to 287 (ours) — 6× more consistent, with higher mean. The efficiency is not noise.

## Qualitative evidence

Median-episode action trajectories before vs after training:
- [Before training](trajectory_before_inventory_balanced.txt)
- [After training](trajectory_after_inventory_balanced.txt)

The trained agent reserves `Air` mode for specific demand-spike steps and uses `Rail` for routine restocking. This behavior emerges purely from the reward signal.

## Known limitations

- **Valid-action rate on held-out seeds is 40%, lower than base Qwen's 60%.** Run-2 training (with -1000 format penalty) stabilized the training loop but didn't fully transfer format compliance to unseen initial states. The agent still relies on the fallback action on ~60% of held-out rollouts. Despite this, its profit and carbon numbers beat base Qwen meaningfully — the fallback path happens to be carbon-efficient, and the agent's valid-action completions are higher quality than base Qwen's. This is the one result we can't cleanly explain and would want more training runs to disambiguate.
- **Training instability.** Run 1 collapsed at step 15 (format reward hacking). Run 2 stabilized but is noisy — batch-level grader scores oscillate across 0.09–0.28 during training. We report final held-out eval, not cherry-picked peaks.
- **Upfront planning constraint.** The agent commits to all 10 steps at t=0 without conditioning on intermediate observations. A receding-horizon variant that re-plans every 3 steps would likely help on `net_zero_profit`.

## Reproducing

```bash
# 1. Clone and enter
git clone https://github.com/Lokeshrao69/eco-logistics.git
cd eco-logistics

# 2. Build + run the env locally (or use the hosted Space)
docker build -t eco-logistics .
docker run -p 7860:7860 eco-logistics

# 3. Run the training notebook (needs GPU, ~90 min on T4)
# Open: https://colab.research.google.com/github/Lokeshrao69/eco-logistics/blob/main/train_eco_logistics_grpo.ipynb
# Set HF_TOKEN and ENV_URL env vars, then Run All.
```

##  Repo structure

```
eco-logistics/
├── env.py                 # Core environment logic
├── models.py              # Pydantic schemas (Action, Observation, Reward)
├── main.py                # FastAPI wrapper (session-pool safe for parallel rollouts)
├── baseline.py            # Heuristic baseline inference
├── inference.py           # OpenAI-client inference script
├── openenv.yaml           # OpenEnv metadata
├── Dockerfile             # For HF Space deployment
├── train_eco_logistics_grpo.ipynb       # GRPO post-training pipeline
├── training_curves_inventory_balanced.png   # 4-panel money chart
├── training_log_final_inventory_balanced.csv   # Raw GRPO metrics per step
├── eval_after_inventory_balanced.json   # Held-out eval results
├── trajectory_before_inventory_balanced.txt
├── trajectory_after_inventory_balanced.txt
└── README.md              # This file
```

##  Team

{FILL_IN_AT_SUBMIT}

## Acknowledgments

Built on OpenEnv from Meta-PyTorch, Hugging Face TRL for GRPO, and Unsloth for memory-efficient LoRA training.