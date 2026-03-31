---
title: Eco-Logistics Supply Chain Optimizer
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
---

# 🌍 Eco-Logistics: Multi-City Supply Chain Optimizer

> **OpenEnv-compliant RL environment** for the Scaler OpenEnv Hackathon.  
> An agent manages inventory across 3 warehouses to meet fluctuating demand while minimizing carbon footprint and shipping costs.

---

## Real-World Motivation

Global supply chains account for **~60% of all carbon emissions** linked to traded goods. Companies like Amazon, Maersk, and Walmart invest billions in logistics optimization — balancing delivery speed against environmental targets.

This environment captures the core tension of modern eco-logistics:

- **Speed vs. Sustainability**: Air freight delivers in 1 step but emits 4x the carbon of rail.
- **Inventory Decay**: Perishable goods lose 2% value per step, modeling spoilage in food/pharma supply chains.
- **Carbon Budgets**: Mirrors real carbon cap-and-trade systems where exceeding limits incurs steep penalties.
- **Weather Disruptions**: Route-specific cost spikes simulate real port closures, storms, and supply shocks.

An agent that solves this environment could be adapted to real warehouse management systems, yielding measurable reductions in both cost and emissions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server (:7860)                │
│  /reset  /step  /state  /tasks  /grader  /baseline      │
└────────────────────────┬────────────────────────────────┘
                         │
                ┌────────▼────────┐
                │  EcoLogisticsEnv │
                │  (env.py)        │
                │                  │
                │  • Demand gen    │
                │  • Shipping sim  │
                │  • Weather events│
                │  • Carbon track  │
                │  • 3 Graders     │
                └────────┬────────┘
                         │
                ┌────────▼────────┐
                │  Pydantic Models │
                │  (models.py)     │
                └─────────────────┘
```

---

## Observation Space

| Field                  | Type                   | Description                                          |
|------------------------|------------------------|------------------------------------------------------|
| `current_inventory`    | `Dict[str, float]`     | Units at each warehouse: Seattle, Chicago, NYC       |
| `pending_shipments`    | `List[PendingShipment]`| In-transit shipments with remaining lead time        |
| `current_demand`       | `Dict[str, float]`     | Customer demand at each city this step                |
| `carbon_credit_balance`| `float`                | Remaining carbon budget (negative = over limit)      |
| `step_number`          | `int`                  | Current step in the episode                          |
| `total_steps`          | `int`                  | Total steps for this task                            |
| `weather_alert`        | `Optional[str]`        | Active weather disruption warning or `null`          |
| `cumulative_profit`    | `float`                | Running profit total                                 |
| `cumulative_carbon`    | `float`                | Running carbon emissions total                       |

## Action Space

| Field              | Type   | Values                            | Description                         |
|--------------------|--------|-----------------------------------|-------------------------------------|
| `ship_amount`      | `float`| `>= 0`                           | Units to ship (0 = no-op)          |
| `origin_city`      | `str`  | `Seattle, Chicago, NYC`           | Source warehouse                    |
| `destination_city`  | `str`  | `Seattle, Chicago, NYC`           | Target warehouse                    |
| `speed_mode`       | `str`  | `Air, Rail`                       | Air: fast/expensive. Rail: slow/cheap |

## Reward Function

```
Reward = Sales_Revenue - Shipping_Cost - Carbon_Penalty - Storage_Fee + Healthy_Stock_Bonus
```

| Component            | Formula                                     |
|----------------------|---------------------------------------------|
| Sales Revenue        | `min(inventory, demand) × $10 per unit`     |
| Shipping Cost        | `route_cost × units × weather_multiplier`   |
| Carbon Penalty       | `carbon_emitted × $1.50 per unit`           |
| Storage Fee          | `total_inventory × $0.50 per unit`          |
| Healthy Stock Bonus  | `+$0.10` if all cities have ≥ 20 units      |

---

## Shipping Routes

| Route              | Rail Cost | Rail Steps | Rail Carbon | Air Cost | Air Steps | Air Carbon |
|--------------------|-----------|------------|-------------|----------|-----------|------------|
| Seattle ↔ Chicago  | $3.00     | 3          | 2.0         | $8.00    | 1         | 8.0        |
| Chicago ↔ NYC      | $2.00     | 2          | 1.5         | $6.00    | 1         | 6.0        |
| Seattle ↔ NYC      | $5.00     | 3          | 3.5         | $12.00   | 1         | 12.0       |

**Weather Events**: Randomly triggered, making a specific route **5× more expensive** for 2 steps. More frequent in volatile demand scenarios (15% per step vs. 5%).

---

## Tasks & Graders

### Task 1: Restock Only (Easy)
- **Goal**: Maintain stock above 20 units at every warehouse for 10 steps.
- **Demand**: Stable (~10 units/city/step).
- **Grader**: `passed_checks / total_checks` — fraction of city-step pairs above 20 units.
- **Carbon Budget**: 200 (generous).

### Task 2: Inventory Balanced (Medium)
- **Goal**: Keep stock levels within 10% of each other across all 3 cities for 15 steps.
- **Demand**: Seasonal (sinusoidal waves).
- **Grader**: `balanced_steps / total_steps` — fraction of steps where max deviation from mean < 10%.
- **Carbon Budget**: 300.

### Task 3: Net-Zero Profit (Hard)
- **Goal**: Maximize profit while keeping total carbon emissions under 80 units over 20 steps.
- **Demand**: Volatile (high variance + weather disruptions).
- **Grader**: Normalized profit score × carbon compliance penalty. Score degrades proportionally to carbon overshoot.
- **Carbon Budget**: 80 (strict).

All graders return a `float` in `[0.0, 1.0]`.

---

## Quick Start

### Docker (Recommended)

```bash
docker build -t eco-logistics .
docker run -p 7860:7860 eco-logistics
```

### Local

```bash
pip install -r requirements.txt
python main.py
```

### API Usage

```bash
# Reset to a task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "restock_only", "seed": 42}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"ship_amount": 10, "origin_city": "Seattle", "destination_city": "NYC", "speed_mode": "Rail"}'

# Check state
curl http://localhost:7860/state

# Grade the episode
curl -X POST http://localhost:7860/grader \
  -H "Content-Type: application/json" \
  -d '{}'

# Run heuristic baseline
curl -X POST http://localhost:7860/baseline \
  -H "Content-Type: application/json" \
  -d '{"task_id": "restock_only"}'
```

### LLM Inference (inference.py)

```bash
# Required environment variables
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="sk-..."

# Start the environment server first
python main.py &

# Run inference (hits all 3 tasks)
python inference.py
```

The inference script uses the **OpenAI Python client** and reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` from environment variables. It connects to the running FastAPI server over HTTP.

---

## Environment Mechanics

### Each Step (in order)
1. **Demand generated** based on task profile (stable/seasonal/volatile).
2. **Weather events** tick down or randomly spawn.
3. **Shipments delivered** if lead time reached zero.
4. **Inventory decay** (2% loss across all warehouses).
5. **Local production** (+20 units per city).
6. **Agent action processed** (shipping queued, cost + carbon charged).
7. **Demand fulfilled** (sell up to available inventory at $10/unit).
8. **Storage fees** charged on remaining inventory.
9. **Reward computed** and returned.

### Key Design Decisions
- **Dense reward**: Every step returns a meaningful signal, not just terminal.
- **Partial progress**: +$0.10 bonus for healthy stock levels encourages survival behavior.
- **Deterministic graders**: Given the same seed + actions, grading is reproducible.
- **Clamped actions**: Over-shipping is silently clamped to available inventory (no crashes).

---

## File Structure

```
eco-logistics/
├── models.py        # Pydantic schemas (Observation, Action, Reward, Tasks)
├── env.py           # Core simulation engine
├── main.py          # FastAPI server
├── baseline.py      # Heuristic baseline agent (used by /baseline endpoint)
├── inference.py     # LLM inference script (OpenAI client, required for submission)
├── openenv.yaml     # OpenEnv specification
├── Dockerfile       # HF Spaces-ready container
├── requirements.txt # Python dependencies
└── README.md        # This file
```

---

## Hugging Face Spaces Deployment

1. Create a new Space with **Docker** SDK.
2. Upload all files.
3. The Dockerfile exposes port **7860** as required.
4. Health check endpoint: `GET /health`.

---

## License

MIT
