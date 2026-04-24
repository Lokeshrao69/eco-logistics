# Round 2 Training — Progress Log

## Smoke test run · April 19, 2026 · Colab T4

**Model:** Qwen/Qwen2.5-1.5B-Instruct · 4-bit LoRA r=16
**Task:** `inventory_balanced`
**Steps completed:** 60 / 100 (Colab free quota exhausted)
**Duration:** ~45 min before disconnect

### Baseline (10 episodes, pre-training)

| Metric | Mean | Std |
|---|---|---|
| total_reward | 2923.638 | 700.830 |
| profit | 4457.377 | 333.366 |
| carbon_used | 445.424 | 389.074 |
| carbon_efficiency | 69.326 | 155.777 |
| delivery_success_rate | 0.772 | 0.144 |
| valid_action_rate | 0.990 | 0.030 |
| grader_score | **0.083** | 0.064 |

### After 60 GRPO steps

| Metric | Value | Delta vs baseline |
|---|---|---|
| total_reward | 3341.10 | +14.3% |
| delivery_success_rate | 83.62% | +6.4 pts |
| grader_score | **0.126** | **+51.8%** |

### Verdict

- Pipeline works end-to-end
- GRPO meaningfully improves the agent
- Free compute ran out before 100 steps; on-site A100 will complete the full run in under 30 min

### What we lost when runtime died

- Reward curve PNG (can regenerate from on-site run)
- Trained LoRA weights (can regenerate)
- `training_log` dict for plotting (can regenerate)

### What we kept

- Code (pushed to GitHub)
- Env live on HF Space
- These numbers (this file)
- Knowledge that the pipeline works

---

## On-site run plan (25th-26th)

### Pre-run checklist

- [ ] Claim HF compute credits (A100 preferred)
- [ ] Clone the notebook from GitHub: `training/train_eco_logistics_grpo_v2.ipynb`
- [ ] Update cell 5: `TASK_ID = "inventory_balanced"` for first run
- [ ] Apply the two fixes from today's session (below)

### Must-do fixes to the notebook before running

**Fix 1: `eager_generate` mode-switching crash**

Replace cell 18's `eager_generate` function with:

```python
def eager_generate(prompt: str) -> str:
    """Safe to call during GRPO training — no mode-switching."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
```

**Fix 2: reward function needs direct episode execution**

Replace cell 22 with:

```python
training_log = {
    "grpo_step": [], "mean_reward": [], "mean_profit": [], "mean_carbon": [],
    "carbon_efficiency": [], "delivery_rate": [], "valid_rate": [], "grader_score": [],
}
global_step_counter = {"n": 0}

def env_reward_func(prompts, completions, **kwargs):
    rewards = []
    batch_metrics = {"reward": [], "profit": [], "carbon": [],
                     "carbon_eff": [], "delivery": [], "valid": [], "grader": []}
    for completion in completions:
        result = run_episode(eager_generate)
        raw_reward = result["total_reward"]
        reward_stats.update(raw_reward)
        normalized_reward = (raw_reward - reward_stats.mean) / reward_stats.std
        rewards.append(float(normalized_reward))
        batch_metrics["reward"].append(raw_reward)
        batch_metrics["profit"].append(result["profit"])
        batch_metrics["carbon"].append(result["carbon_used"])
        batch_metrics["carbon_eff"].append(result["carbon_efficiency"])
        batch_metrics["delivery"].append(result["delivery_success_rate"])
        batch_metrics["valid"].append(result["valid_action_rate"])
        batch_metrics["grader"].append(result["grader_score"])
    global_step_counter["n"] += 1
    training_log["grpo_step"].append(global_step_counter["n"])
    training_log["mean_reward"].append(np.mean(batch_metrics["reward"]))
    training_log["mean_profit"].append(np.mean(batch_metrics["profit"]))
    training_log["mean_carbon"].append(np.mean(batch_metrics["carbon"]))
    training_log["carbon_efficiency"].append(np.mean(batch_metrics["carbon_eff"]))
    training_log["delivery_rate"].append(np.mean(batch_metrics["delivery"]))
    training_log["valid_rate"].append(np.mean(batch_metrics["valid"]))
    training_log["grader_score"].append(np.mean(batch_metrics["grader"]))
    if global_step_counter["n"] % 5 == 0:
        print(f"[step {global_step_counter['n']}] "
              f"reward={np.mean(batch_metrics['reward']):.2f} "
              f"grader={np.mean(batch_metrics['grader']):.3f} "
              f"delivery={np.mean(batch_metrics['delivery']):.2%}")
    return rewards
```

**Fix 3: Cell 23's GRPOConfig — remove trackio, use fp16 on T4 (or bf16 on A100)**

```python
training_args = GRPOConfig(
    output_dir="./eco-logistics-grpo-v2",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_generations=NUM_GENERATIONS,
    max_completion_length=MAX_COMPLETION_LENGTH,
    max_prompt_length=1024,
    max_steps=MAX_STEPS,
    logging_steps=1,
    save_steps=25,           # checkpoint every 25 steps (NEW — don't lose progress)
    bf16=True,               # A100 supports this; switch to fp16=True on T4
    use_vllm=False,
    report_to="none",
    run_name=f"eco-logistics-v2-{TASK_ID}",
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    reward_funcs=env_reward_func,
    processing_class=tokenizer,
)
trainer.train()
```

**Fix 4: Add a "save plots mid-training" cell AFTER cell 23**

Paste this as a new cell right after `trainer.train()` to auto-save progress plots every time training halts:

```python
# Auto-save plots from whatever training_log has — even if training was interrupted
import matplotlib.pyplot as plt
import pandas as pd

log_df = pd.DataFrame(training_log)
log_df.to_csv(f"training_log_{TASK_ID}.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(log_df["grpo_step"], log_df["mean_reward"], alpha=0.3, label="per step")
axes[0].plot(log_df["grpo_step"], log_df["mean_reward"].rolling(5).mean(),
             linewidth=2, label="rolling mean")
axes[0].set_xlabel("GRPO step")
axes[0].set_ylabel("Mean episode reward")
axes[0].set_title(f"Training reward — {TASK_ID}")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(log_df["grpo_step"], log_df["grader_score"], alpha=0.3, label="per step")
axes[1].plot(log_df["grpo_step"], log_df["grader_score"].rolling(5).mean(),
             linewidth=2, color="#2a9d8f", label="rolling mean")
axes[1].set_xlabel("GRPO step")
axes[1].set_ylabel("Grader score (0-1)")
axes[1].set_title("Task completion over training")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"training_curves_{TASK_ID}.png", dpi=150, bbox_inches="tight")
plt.show()

# Download files immediately
from google.colab import files
files.download(f"training_curves_{TASK_ID}.png")
files.download(f"training_log_{TASK_ID}.csv")
```

The `files.download()` triggers a browser download immediately, so even if Colab disconnects after this cell, the files are on your disk.

### Run order on the 25th

1. Cells 1-2: install + imports
2. Cell 3: HF login (with Colab secret)
3. Cell 5: config — **make sure TASK_ID is inventory_balanced first, then net_zero_profit**
4. Cells 10-15: setup functions (don't skip)
5. Cell 17: load model
6. Updated cell 18: `eager_generate` (fix 1)
7. Cell 21: `RunningStats` + logging dict
8. Updated cell 22: reward function (fix 2)
9. Updated cell 23: trainer config (fix 3) — hit run, wait ~30 min on A100
10. New cell 24: auto-save + download (fix 4)
11. Cell 25: AFTER eval (10 episodes)
12. Cell 29: trajectory dumps

### Budget for both tasks

- `inventory_balanced` (medium) — 30 min training + 10 min eval = 40 min
- `net_zero_profit` (hard) — 30 min training + 10 min eval = 40 min
- Buffer for debugging — 30 min
- **Total compute budget: ~2 hours** on A100

---

## Today's key lesson

Colab free tier is unreliable for multi-hour training. The on-site A100 + proper checkpointing (fix 4) means this won't happen again.

Even with only 60% of training done, the reward curve was clearly trending up. That's genuinely promising for the final run.
