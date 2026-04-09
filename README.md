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

> An OpenEnv-compatible reinforcement learning environment where an AI agent manages inventory across 3 warehouses to meet fluctuating demand while minimizing carbon footprint and shipping costs.

---

## What is Eco-Logistics?

Eco-Logistics simulates the core challenge of modern green supply chain management — balancing delivery speed, cost efficiency, and environmental sustainability across a multi-city warehouse network.

### Real-World Applications

- **Freight logistics** — choosing between air and rail shipments based on urgency vs. carbon cost
- **Warehouse management** — pre-positioning inventory to meet regional demand spikes
- **Carbon cap-and-trade** — operating under strict emissions budgets while maintaining profitability
- **Perishable goods** — managing inventory decay (food, pharma, chemicals)
- **Disaster response** — rerouting around disrupted corridors (modeled as weather events)

### Why This Matters

Global supply chains account for ~60% of carbon emissions linked to traded goods. Companies like Amazon, Maersk, and Walmart invest billions optimizing this exact tradeoff. An agent that solves this environment demonstrates skills directly transferable to real warehouse management systems.

---

## Key Terms

| Term | Description |
|------|-------------|
| **Warehouse** | One of 3 cities: Seattle, Chicago, NYC. Each has local inventory. |
| **Shipment** | Transfer of goods between warehouses. Has cost, lead time, and carbon emissions. |
| **Lead time** | Steps until a shipment arrives. Rail: 2-3 steps. Air: 1 step. |
| **Inventory decay** | 2% of all inventory lost each step (models perishable goods). |
| **Carbon budget** | Total allowed CO₂ emissions for the episode. Exceeding it penalizes score. |
| **Weather event** | Random disruption that makes a specific route 5× more expensive for 2 steps. |
| **Demand profile** | How customer demand fluctuates: stable, seasonal, volatile, or surge. |
| **No-op** | Setting `ship_amount=0` — a valid action meaning "do nothing this step." |

---

## Environment Overview

```
Agent  ──→  EcoLogisticsEnv  ──→  FastAPI Server (Docker, port 7860)
  │              │
  │  reset()     │  Initializes task, returns Observation
  │  step()      │  Processes action, returns (Observation, Reward, done, info)
  │  state()     │  Returns full environment snapshot
  │  grade()     │  Scores the episode (0.0–1.0)
```

### Simulation Loop (each step, in order)

1. **Demand generated** — based on task profile (stable / seasonal / volatile / surge)
2. **Weather events** — tick down active events or randomly spawn new ones
3. **Shipments delivered** — arrived goods added to destination inventory
4. **Inventory decay** — all warehouses lose 2% of stock
5. **Local production** — each city produces +20 units
6. **Agent action processed** — shipment queued, cost and carbon charged
7. **Demand fulfilled** — sell up to available inventory at $10/unit
8. **Storage fees** — charged on all remaining inventory
9. **Reward computed** — dense signal returned to agent

---

## Action Space

```python
class Action(BaseModel):
    ship_amount: float      # Units to ship (0 = no-op)
    origin_city: str        # "Seattle", "Chicago", or "NYC"
    destination_city: str   # "Seattle", "Chicago", or "NYC"
    speed_mode: SpeedMode   # "Air" or "Rail"
```

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `ship_amount` | `float` | `≥ 0` | Units to ship. 0 means do nothing. Clamped to available inventory. |
| `origin_city` | `str` | `Seattle, Chicago, NYC` | Source warehouse |
| `destination_city` | `str` | `Seattle, Chicago, NYC` | Target warehouse (must differ from origin) |
| `speed_mode` | `str` | `Air, Rail` | Air: fast (1 step), expensive, high carbon. Rail: slow (2-3 steps), cheap, low carbon. |

### Example Actions

```python
# Ship 15 units Seattle → NYC by Rail
Action(ship_amount=15, origin_city="Seattle", destination_city="NYC", speed_mode="Rail")

# Do nothing this step
Action(ship_amount=0, origin_city="Seattle", destination_city="Chicago", speed_mode="Rail")
```

---

## Observation Space

```python
class Observation(BaseModel):
    current_inventory: Dict[str, float]        # Stock at each warehouse
    pending_shipments: List[PendingShipment]    # In-transit goods
    current_demand: Dict[str, float]            # Customer demand this step
    carbon_credit_balance: float                # Remaining carbon budget
    step_number: int                            # Current step
    total_steps: int                            # Episode length
    weather_alert: Optional[str]               # Active disruption warning
    cumulative_profit: float                    # Running profit total
    cumulative_carbon: float                    # Running emissions total
```

| Field | Type | Description |
|-------|------|-------------|
| `current_inventory` | `Dict[str, float]` | Units at Seattle, Chicago, NYC |
| `pending_shipments` | `List[PendingShipment]` | Each has origin, destination, amount, steps_remaining, speed_mode |
| `current_demand` | `Dict[str, float]` | Customer demand at each city this step |
| `carbon_credit_balance` | `float` | Budget remaining. Negative = over limit. |
| `step_number` | `int` | Current step (0-indexed) |
| `total_steps` | `int` | Total steps in this episode |
| `weather_alert` | `Optional[str]` | E.g. `"Chicago→NYC route cost is 5.0x for 2 more step(s)."` or `null` |
| `cumulative_profit` | `float` | Running total profit |
| `cumulative_carbon` | `float` | Running total emissions |

---

## Shipping Routes

| Route | Rail Cost/unit | Rail Lead Time | Rail Carbon/unit | Air Cost/unit | Air Lead Time | Air Carbon/unit |
|-------|---------------|----------------|------------------|---------------|---------------|-----------------|
| Seattle ↔ Chicago | $3.00 | 3 steps | 2.0 | $8.00 | 1 step | 8.0 |
| Chicago ↔ NYC | $2.00 | 2 steps | 1.5 | $6.00 | 1 step | 6.0 |
| Seattle ↔ NYC | $5.00 | 3 steps | 3.5 | $12.00 | 1 step | 12.0 |

**Weather events** randomly make a route **5× more expensive** for 2 steps. Frequency: 5% per step (stable/seasonal) or 15% per step (volatile/surge).

---

## Reward Function

```
Reward = Sales_Revenue - Shipping_Cost - Carbon_Penalty - Storage_Fee + Healthy_Stock_Bonus
```

| Component | Formula | Signal |
|-----------|---------|--------|
| Sales Revenue | `min(inventory, demand) × $10/unit` | Positive — fulfill demand |
| Shipping Cost | `route_cost × units × weather_multiplier` | Negative — shipping is expensive |
| Carbon Penalty | `carbon_emitted × $1.50/unit` | Negative — emissions cost money |
| Storage Fee | `total_inventory × $0.50/unit` | Negative — holding stock costs money |
| Healthy Stock Bonus | `+$0.10` if all cities ≥ 20 units | Positive — partial progress signal |

The reward is **dense** — every step returns a meaningful, varying signal (verified: 20 unique values across 20 steps).

---

## Tasks & Graders

### Task 1 — Restock Only (Easy)

| Property | Value |
|----------|-------|
| Steps | 10 |
| Demand | Stable (~10 units/city/step) |
| Carbon budget | 200 (generous) |
| Initial inventory | 50/50/50 |
| Grader | Fraction of city-step checks where inventory ≥ 20 |
| Strategy hint | Do nothing — local production (+20/step) handles demand (~10/step) |

### Task 2 — Inventory Balanced (Medium)

| Property | Value |
|----------|-------|
| Steps | 15 |
| Demand | Seasonal (sinusoidal waves) |
| Carbon budget | 300 |
| Initial inventory | 60/40/80 (deliberately unbalanced) |
| Grader | 60% perfect-balance ratio + 40% average closeness (smooth gradient) |
| Strategy hint | Ship from NYC (80) to Chicago (40) via Rail to equalize |

### Task 3 — Net-Zero Profit (Hard)

| Property | Value |
|----------|-------|
| Steps | 20 |
| Demand | Volatile (high variance + weather) |
| Carbon budget | 80 (very strict) |
| Initial inventory | 40/40/40 |
| Grader | Normalized profit × carbon compliance penalty |
| Strategy hint | Mostly no-op. Every shipment burns carbon. Only ship Rail when critical. |

### Task 4 — Demand Surge Response (Hard)

| Property | Value |
|----------|-------|
| Steps | 15 |
| Demand | NYC demand spikes to 3× after step 7 |
| Carbon budget | 150 |
| Initial inventory | 30/30/30 |
| Grader | 70% demand fulfillment rate + 30% carbon compliance |
| Strategy hint | Pre-position inventory in NYC before step 7. Use Rail to save carbon. |

All graders return scores strictly in `(0.0, 1.0)`.

---

## Baseline Scores

Results with `seed=42`:

| Strategy | Restock Only | Inventory Balanced | Net-Zero Profit | Demand Surge |
|----------|:---:|:---:|:---:|:---:|
| No-op | 0.999 | 0.201 | 0.465 | 0.965 |
| Aggressive Air | 0.800 | 0.003 | 0.001 | 0.615 |
| Moderate Rail | 0.999 | 0.012 | 0.053 | 0.999 |
| Heuristic baseline | 0.999 | 0.701 | 0.001 | — |

---

## Project Structure

```
eco-logistics/
├── models.py               # Pydantic schemas (Observation, Action, Reward, Tasks)
├── env.py                  # Core simulation engine (step/reset/state/grade)
├── main.py                 # FastAPI server (all endpoints)
├── baseline.py             # Heuristic baseline agent
├── inference.py            # LLM inference script (OpenAI client, structured logs)
├── test_env.py             # Automated test suite (76 tests)
├── validate-submission.sh  # Pre-submission validation script
├── openenv.yaml            # OpenEnv specification
├── pyproject.toml          # Project metadata and dependencies
├── uv.lock                 # Locked dependencies
├── requirements.txt        # Pip dependencies
├── Dockerfile              # HF Spaces container (port 7860)
├── README.md               # This file
└── server/
    ├── __init__.py
    └── app.py              # OpenEnv server entry point
```

---

## Setup & Usage

### Prerequisites

- Python ≥ 3.10
- Docker (for deployment)

### Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Tests

```bash
python test_env.py           # 76 automated tests
bash validate-submission.sh  # Full pre-submission check
```

### Start Server

```bash
python main.py
# Server runs on http://localhost:7860
# Interactive docs: http://localhost:7860/docs
```

### API Usage

```bash
# List tasks
curl http://localhost:7860/tasks

# Reset to a task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"restock_only","seed":42}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"ship_amount":10,"origin_city":"Seattle","destination_city":"NYC","speed_mode":"Rail"}'

# Grade the episode
curl -X POST http://localhost:7860/grader
```

### Run LLM Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="hf_..."
python inference.py
```

Expected output format:
```
[START] task=restock_only env=eco_logistics model=meta-llama/Llama-3.3-70B-Instruct
[STEP] step=1 action=no-op reward=153.40 done=false error=null
[STEP] step=2 action=ship(10,Seattle->NYC,Rail) reward=141.20 done=false error=null
...
[END] success=true steps=10 score=0.999 rewards=153.40,141.20,...
```

### Docker

```bash
docker build -t eco-logistics .
docker run -p 7860:7860 eco-logistics
```

---

## Hugging Face Space

The environment is deployed at: https://huggingface.co/spaces/lokeshrao226/eco-logistics

| Endpoint | Description |
|----------|-------------|
| `GET /` | Environment info and endpoint list |
| `GET /health` | Health check |
| `GET /tasks` | List all 4 tasks with descriptions |
| `GET /state` | Full environment state snapshot |
| `POST /reset` | Reset to a task (accepts empty body) |
| `POST /step` | Execute one step |
| `POST /grader` | Grade the current episode |
| `POST /baseline` | Run heuristic baseline |
| `GET /docs` | Interactive Swagger documentation |

---

## License

MIT