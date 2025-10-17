# Auralis
An adaptive energy optimization sandbox with controllers, safety constraints, economic KPIs, and product surfaces (API + dashboard). Start with a single node, scale to multi-node with shared constraints, and export artifacts for analysis.

## Quickstart

```bash
pip install -r requirements.txt

# Run controllers and plots
python simulation.py

# API (Swagger at /docs)
uvicorn api:app --reload

# Dashboard (Streamlit)
streamlit run dashboard.py
```

## Features
- Single- and multi-node simulation with safety constraints (energy bounds, temperature cap)
- Controllers: Heuristic, Rule-based, MPC-lite, CVXPY MPC
- Economic KPIs (baseline/control cost, savings) and technical KPIs (stability, violations)
- Artifact saving (CSV/JSON) per run + plots
- REST API (FastAPI) + Dashboard (Streamlit)
- Tests, typing, linting, CI (GitHub Actions)

## Results (example)
Run a small leaderboard across seeds:

```python
from simulation import run_leaderboard
run_leaderboard()
```

Expected output (example):

```
=== Leaderboard (mean over seeds) ===
HeuristicController    savings=$X.XX  violations=Y
RuleBasedController    savings=$X.XX  violations=Y
MPCLiteController      savings=$X.XX  violations=Y
CvxpyMPCController     savings=$X.XX  violations=Y
```

## Repo layout
- `Nodes/base_node.py`: Node physics, logging, constraints
- `simulation.py`: World, controllers, runners, multi-node
- `utils.py`: Plotting, KPIs, artifacts
- `api.py`: FastAPI service
- `dashboard.py`: Streamlit app
- `sdk.py`: Thin SDK for suggestions and runs
- `tests/`: Pytest suite

## License
MIT
