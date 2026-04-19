# Eco-Logistics GRPO Training Plan — Round 2

This folder contains the training script and plan for Round 2 of the Scaler OpenEnv Hackathon.

## What we're doing

Post-training a small LLM (Qwen2.5-1.5B-Instruct) to act as a supply-chain agent inside the `eco-logistics` OpenEnv environment we built in Round 1. We use **GRPO** (Group Relative Policy Optimization) via HuggingFace TRL, with **Unsloth + LoRA** to fit training on a single Colab GPU.

**Theme fit:** Primary — Theme #3.1 (World Modeling / Professional Tasks). Secondary — Theme #2 (Long-Horizon Planning) for the Net-Zero Profit task.

## Files

- `train_eco_logistics_grpo.ipynb` — the full training notebook, Colab-ready

## Pre-onsite checklist (do these before the 25th)

### 1. Add concurrency support to the HF Space

The current `main.py` in your `eco-logistics` repo probably uses `create_app(...)` without `max_concurrent_envs`. GRPO will fire 4+ parallel rollouts and your env will 429. Fix:

```python
# in main.py
app = create_app(
    create_eco_logistics_env,
    Action,
    Observation,
    max_concurrent_envs=8,   # must be >= per_device_train_batch_size * gradient_accumulation_steps
)
```

Test locally, push to the Space, confirm it still validates.

### 2. Dry-run the smoke-test cell

Open the notebook in Colab on a free T4. Run cells 1-3 (setup, config, smoke-test) and confirm the `/reset` and `/step` calls come back with your actual schema. Fix any key-name mismatches between `format_observation_prompt` and the real observation dict your env returns.

### 3. Run 10 baseline episodes

Skip the training cell, run cells 1-6 plus cell 15 (eager_generate). Print the baseline reward and grader score. Write these numbers down — they're your "before" numbers for the pitch.

### 4. Decide: Qwen2.5-1.5B vs 3B

1.5B trains fast but may struggle with JSON format. 3B is slower but cleaner outputs. Test both in the smoke-test phase and pick whichever has a higher `valid_action_rate` on the baseline.

## On-site plan (25th-26th)

### Day 1 (25th)

- **Morning:** Claim compute credits, load the notebook on the provided instance (A100 ideally)
- **Hour 1-2:** Run baseline eval on all 3 tasks, save numbers
- **Hour 2-6:** Train on `restock_only` (easy — should show a curve in 50 steps)
- **Hour 6-10:** Train on `inventory_balanced` (medium)

### Day 2 (26th)

- **Morning:** Train on `net_zero_profit` (hard — this is your money-shot reward curve)
- **Midday:** Generate all plots, capture 2 trajectories per task (before/after qualitative examples)
- **Afternoon:** Record the <2min video, write the HF mini-blog
- **Evening:** Rehearse the 3-minute pitch

## Pitch narrative sketch

> "Carbon-aware logistics is a $50B problem where every shipping decision trades cost against emissions. Today's LLMs can describe the problem but can't *act* on it — they hallucinate shipping actions or ignore the carbon budget entirely. We built Eco-Logistics, a partially-observable supply-chain environment with dense reward signals and three escalating tasks. Then we post-trained a 1.5B-parameter model inside it using GRPO. Here's the reward curve [SHOW CURVE]. The baseline model scores X on our hardest task; after 300 GRPO steps, it scores Y — a Z% improvement, with a fraction of GPT-4o's parameters."

The key beats for the 30% storytelling score: **real problem → clean env → observable training signal → concrete delta**.

## Known risks + mitigations

| Risk | Mitigation |
|------|------------|
| vLLM import breaks on Colab | Start with `use_vllm=False`, switch on A100 only after baseline confirmed |
| Model emits non-JSON | `parse_action` already falls back to safe no-op; valid_action_rate becomes a secondary metric that itself improves with training |
| Reward curve is flat on `net_zero_profit` | Pre-train briefly on `restock_only` first, then warm-start on harder tasks |
| HF Space cold-start timeouts during training | Bump smoke-test timeout to 60s, add a retry wrapper around the `requests.post` calls in `run_episode` |
| Colab disconnects mid-training | Checkpoint every 50 steps (already configured); resume from `./eco-logistics-grpo/checkpoint-*` |

## Hyperparameters to tune if reward curve is flat

1. **`num_generations`** — bump from 4 → 8. More samples per group = lower-variance advantage estimates.
2. **`learning_rate`** — try 1e-5 if nothing is moving; drop to 2e-6 if it's unstable.
3. **`beta` (KL penalty)** — default 0.04 in TRL. Lower to 0.01 for more exploration on hard tasks.
4. **Reward shaping** — if dense env reward has huge variance, normalize by dividing each episode return by its running std. TRL does this internally but watch for exploding rewards on high-carbon runs.
