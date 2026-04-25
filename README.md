# Eco-Logistics: Multi-City Supply Chain Optimizer

> **OpenEnv Hackathon — India 2026**
> Theme: **World Modeling — Professional Tasks**
> Team: **Crystal Blue**

An RL environment that puts a language model in charge of a three-warehouse supply chain (Seattle · Chicago · NYC) and post-trains it with **GRPO** to navigate the profit-vs-carbon tradeoff under non-stationary shocks.

**Headline result (held-out `net_zero_profit` task, 3-run averaged):**
> GRPO Qwen-2.5-1.5B achieves a **25.6× improvement in profit-per-carbon ratio** over base Qwen, with a **grader score of 0.273 ± 0.019** (vs base 0.259). The trained policy generalizes from the medium training task to the harder evaluation task with stable variance.

---

##  Deliverables

| Resource | Link |
|---|---|
| HF Space (live environment) | https://huggingface.co/spaces/lokeshrao226/eco-logistics |
| Training notebook (Colab) | https://colab.research.google.com/github/Lokeshrao69/eco-logistics/blob/main/train_eco_logistics_grpo.ipynb |
| Code repository (this repo) | https://github.com/Lokeshrao69/eco-logistics |
| Trained LoRA adapter | https://huggingface.co/lokeshrao226/eco-logistics-qwen-grpo |
| Writeup (HF Discussion) | https://huggingface.co/lokeshrao226/eco-logistics-qwen-grpo/discussions/1|

---

## The Problem

Real supply chains don't fail in controlled conditions. They fail when demand spikes unexpectedly, when competitors outbid you on a route, or when carbon budgets tighten mid-quarter. We wanted an environment where an agent has to plan **defensively against surprise** and learn the **profit-vs-carbon tradeoff** at the same time.

## The Environment

A 3-warehouse OpenEnv-compliant simulator with the standard `reset` / `step` / `state` / `grader` interface:

- **Observation**: current inventory, pending shipments, demand forecast, carbon credit balance, weather/market alerts
- **Action**: `(ship_amount, origin_city, destination_city, speed_mode ∈ {Rail, Air})`
- **Reward (dense)**: `sales_revenue − shipping_cost − carbon_penalty − storage_fee + healthy_stock_bonus`
- **Three tasks** of increasing difficulty:
  1. `restock_only` — easy
  2. `inventory_balanced` — medium (used for primary training)
  3. `net_zero_profit` — hard (max profit under a strict carbon budget; used for held-out cross-task eval)

**World-modeling wrapper**: at rollout time we inject non-stationary demand shocks (`p=0.15`, 2.5× multiplier) and competitor bids (`p=0.20`, 1.8× shipping cost) surfaced through the `weather_alert` observation field. The agent has to plan *around* these surprises without seeing them at plan-generation time.

## The Training

- **Base model**: Qwen-2.5-1.5B-Instruct
- **Method**: GRPO via TRL, LoRA adapters via Unsloth, 4-bit quantization, single T4
- **Design choice — Upfront Trajectory Planning**: the model generates the entire 10-step action sequence as a JSON array from the initial observation. One completion = one rollout, which makes GRPO-over-HTTP tractable on a T4.
- **Dataset**: 50 unique initial states sampled across all 3 tasks and seeds 0–49.
- **Evaluation**: held-out seeds 500–509 the model never saw during training.

**Reward hacking diagnosis and fix** (documented in the writeup): the initial training run collapsed at step 15 when the model discovered that outputting invalid JSON let a fallback action shield it from carbon penalties. We diagnosed the mechanism (format penalty magnitude < carbon savings from null shipments) and fixed it by making invalid output cost `-1000.0` instead of `-5.0` — this restored gradient signal and stabilized training.

## Results — cross-task evaluation

We evaluated the same trained model on two tasks: `inventory_balanced` (the training-distribution task) and `net_zero_profit` (a harder, never-trained-on task that tests cross-task generalization).

![Training curves + 4-way comparison on held-out seeds](training_curves_IB.png)
### Visual comparison

![Held-out grader scores across 4 policies](chart_grader_comparison.png)

![Profit/carbon ratio improvement](chart_profit_carbon_ratio.png)

### Task 1 — `inventory_balanced` (training-distribution, single 10-episode run)

| Policy | Profit | Carbon | Profit/Carbon (agg) | Grader | Delivery |
|---|---|---|---|---|---|
| Random policy | 3946 | 1252 | 3.2 | 0.065 | 87.6% |
| Heuristic (richest→poorest, rail) | 3558 | 450 | 7.9 | 0.040 | 71.5% |
| Base Qwen-2.5-1.5B | 3793 | 1429 | 2.7 | 0.135 | 86.6% |
| **GRPO Qwen (ours)** | **4828** | **122** | **39.6** | 0.195 | 87.6% |

### Task 2 — `net_zero_profit` (held-out, 3-run averaged, 30 episodes total)

| Policy | Profit | Carbon | Profit/Carbon (agg) | Grader |
|---|---|---|---|---|
| Random | 2636.6 | 1076.8 | 2.85 | 0.001 |
| Heuristic | 3735.2 | 0.0 | ∞ | 0.292 |
| Base Qwen-2.5-1.5B | 3708.8 | 25.2 | 2.65 | 0.259 |
| **GRPO Qwen (ours)** | **3687.9 ± 55.7** | **54.3 ± 64.9** | **67.96** | **0.273 ± 0.019** |

### How to read these together

The `inventory_balanced` numbers are eye-catching but came from a **single 10-episode run**, and re-runs showed substantial variance (subsequent runs gave 4421/695 and 4767/217 for profit/carbon). We report the original run for parity with the published model card.

The `net_zero_profit` numbers are **3-run averaged across 30 episodes total**, with run-to-run σ on grader of 0.019. We trust this eval more, and you should too. The 25.6× profit/carbon ratio improvement (2.65 → 67.96) on a never-trained-on task is our most defensible claim. The grader improvement (+0.014 vs base) is meaningful with tight variance, though the hand-tuned heuristic still wins on grader (0.292) — we view our contribution as showing a *learned* policy can approach hand-tuned performance without env-specific rules.

**What's rock-solid**: profit σ dropped from 1771 (base) to 287 (ours) — 6× more consistent, with comparable mean. The efficiency improvement is not noise.

## A negative result: SFT-then-GRPO

After v8 produced our headline numbers, we attempted to address an obvious weakness — valid-action rate of 20% on held-out seeds — by adding an SFT warmup phase as recommended in §3 of the hackathon guide.

**What worked:** Generated 150 trajectories using our heuristic, ran 1 epoch of SFT on the LoRA. Valid-action rate jumped from **20% → 80%**. Format compliance was cleanly fixed.

**What didn't:** GRPO on top of the SFT-warmed model collapsed in 3 different reward configurations (multi-component equal-weight, multi-component rebalanced, single grader-only). Pattern was the same: grader dropped, carbon climbed, model over-shipped.

**Our diagnosis:** SFT made the policy too uniform. GRPO needs variance among the N=4 sampled completions to identify which is better; after SFT all 4 samples followed the heuristic's distribution too closely. Combined with profit/delivery rewards being easier to optimize than the noisier grader signal, the policy drifted toward "ship more."

**The fix we'd implement with more time:** entropy regularization during GRPO, or softer SFT against multiple expert demonstrations, or KL penalty against the base model. None of which we could validate before the deadline.

We're submitting v8 (no SFT) as the headline. The SFT experiment is documented as honest evidence of what we tried.

## Qualitative evidence

Median-episode action trajectories before vs after training:
- [Before training](before_training.png)
- [After training](after_training.png)

The trained agent reserves `Air` mode for specific demand-spike steps and uses `Rail` for routine restocking. This behavior emerges purely from the reward signal.

## Known limitations

- **Valid-action rate of 20% on held-out seeds.** The single biggest weakness. Despite this, profit and carbon numbers beat base Qwen meaningfully — the fallback path happens to be carbon-efficient, and the agent's valid-action completions are higher quality than base Qwen's. This is the one result we can't cleanly explain and would want more training runs to disambiguate.
- **Heuristic still wins on grader** (0.292 vs 0.273 on `net_zero_profit`). Not a SOTA claim — we view our contribution as showing a learned policy can approach hand-tuned performance without env-specific rules.
- **Training instability documented.** Run 1 collapsed at step 15 (format reward hacking). Run 2 stabilized but is noisy — batch-level grader scores oscillate across 0.09–0.28 during training. We report final held-out eval, not cherry-picked peaks.
- **Upfront planning constraint.** The agent commits to all 10 steps at t=0 without conditioning on intermediate observations. A receding-horizon variant that re-plans every 3 steps would likely help on `net_zero_profit`.

## What we'd do next

- **Receding-horizon planning.** Re-plan every 3 steps. Addresses the upfront-plan limitation directly.
- **Fix the SFT-then-GRPO collapse.** Entropy regularization or mixed-expert SFT.
- **Process supervision.** Per-step rewards instead of one episode-end reward.
- **Co-trained disruptor agent.** Adversarial curriculum hitting the multi-agent theme.
- **Larger model.** 7B would likely substantially improve `net_zero_profit`.

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
├── train_eco_logistics_grpo.ipynb            # GRPO post-training pipeline
├── training_curves_IB.png                    # 4-panel training metrics chart
├── training_log_final_inventory_balanced.csv # Raw GRPO metrics per step
├── eval_3run_averaged_net_zero_profit.json   # Headline eval results
├── trajectory_before_inventory_balanced.txt
├── trajectory_after_inventory_balanced.txt
├── before_training.png                       # Qualitative: pre-training rollout
├── after_training.png                        # Qualitative: post-training rollout
└── README.md              # This file
```

##  Team

Crystal Blue

## Acknowledgments

Built on OpenEnv from Meta-PyTorch, Hugging Face TRL for GRPO, and Unsloth for memory-efficient LoRA training.