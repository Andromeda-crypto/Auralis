from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from sdk import run_controller, run_multinode, suggest_setpoint

app = FastAPI(title="Auralis API")

@app.get("/")
def root() -> dict:
    return {
        "status": "ok",
        "message": "Auralis API is running. See /docs for interactive docs.",
        "endpoints": ["/suggest", "/run", "/run_multi", "/docs"],
    }

# --- Suggest endpoint --- #
class SuggestRequest(BaseModel):
    energy: float
    capacity: float
    price: float
    supply: float
    demand: float
    controller: Optional[str] = "rule_based"

@app.post("/suggest")
def suggest(req: SuggestRequest) -> dict:
    state = req.dict()
    ctrl = state.pop("controller", "rule_based")
    return suggest_setpoint(state, controller_name=ctrl)

# --- Single-node run --- #
class RunRequest(BaseModel):
    controller: Optional[str] = "rule_based"
    steps: int = 60
    seed: int = 123

@app.post("/run")
def run(req: RunRequest) -> dict:
    try:
        res = run_controller(
            controller_name=str(req.controller),
            steps=req.steps,
            seed=req.seed,
            do_plots=False,
        )
        return {
            "history": res.get("history", []),
            "prices": res.get("prices", []),
            "baseline_import": res.get("baseline_import", []),
            "control_import": res.get("control_import", []),
            "kpis": res.get("kpis", {}),
            "econ": res.get("econ", {}),
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/run")
def run_get(
    controller: Optional[str] = "rule_based",
    steps: int = 60,
    seed: int = 123,
) -> dict:
    try:
        res = run_controller(
            controller_name=str(controller),
            steps=steps,
            seed=seed,
            do_plots=False,
        )
        return {
            "history": res.get("history", []),
            "prices": res.get("prices", []),
            "baseline_import": res.get("baseline_import", []),
            "control_import": res.get("control_import", []),
            "kpis": res.get("kpis", {}),
            "econ": res.get("econ", {}),
        }
    except Exception as e:
        return {"error": str(e)}

# --- Multi-node run --- #
class MultiRunRequest(BaseModel):
    controller: Optional[str] = "rule_based"
    num_nodes: int = 3
    feeder_limit: float = 0.8
    steps: int = 60
    seed: int = 2024

@app.post("/run_multi")
def run_multi(req: MultiRunRequest) -> dict:
    try:
        res = run_multinode(
            num_nodes=req.num_nodes,
            feeder_limit=req.feeder_limit,
            steps=req.steps,
            seed=req.seed,
            controller_name=str(req.controller),
            do_plots=False,
        )

        comparison = [
            {"node": i + 1, "baseline": base, "controlled": ctrl}
            for i, (base, ctrl) in enumerate(zip(res["site_baseline_import"], res["site_control_import"]))
        ]
        reduction = round(
            100 * (sum(res["site_baseline_import"]) - sum(res["site_control_import"]))
            / sum(res["site_baseline_import"]),
            2,
        )
        return {"comparison": comparison, "reduction": reduction}
    except Exception as e:
        return {"error": str(e)}

@app.get("/run_multi")
def run_multi_get(
    controller: Optional[str] = "rule_based",
    num_nodes: int = 3,
    feeder_limit: float = 0.8,
    steps: int = 60,
    seed: int = 2024,
) -> dict:
    try:
        res = run_multinode(
            num_nodes=num_nodes,
            feeder_limit=feeder_limit,
            steps=steps,
            seed=seed,
            controller_name=str(controller),
            do_plots=False,
        )
        return {
            "site_baseline_import": res["site_baseline_import"],
            "site_control_import": res["site_control_import"],
        }
    except Exception as e:
        return {"error": str(e)}

