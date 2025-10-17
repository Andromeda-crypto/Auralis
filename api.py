from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from sdk import suggest_setpoint, run_controller, run_multinode

app = FastAPI(title="Auralis API")

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Auralis API is running. See /docs for interactive docs.",
        "endpoints": ["/suggest", "/run", "/run_multi", "/docs"],
    }


class SuggestRequest(BaseModel):
    energy: float
    capacity: float
    price: float
    supply: float
    demand: float
    controller: Optional[str] = "rule_based"


@app.post("/suggest")
def suggest(req: SuggestRequest):
    state = req.dict()
    ctrl = state.pop("controller", "rule_based")
    return suggest_setpoint(state, controller_name=ctrl)


class RunRequest(BaseModel):
    controller: Optional[str] = "rule_based"
    steps: int = 60
    seed: int = 123


@app.post("/run")
def run(req: RunRequest):
    res = run_controller(controller_name=req.controller, steps=req.steps, seed=req.seed, do_plots=False)
    return {"kpis": res["kpis"], "econ": res["econ"]}


class MultiRunRequest(BaseModel):
    controller: Optional[str] = "rule_based"
    num_nodes: int = 3
    feeder_limit: float = 0.8
    steps: int = 60
    seed: int = 2024


@app.post("/run_multi")
def run_multi(req: MultiRunRequest):
    res = run_multinode(num_nodes=req.num_nodes, feeder_limit=req.feeder_limit, steps=req.steps, seed=req.seed, controller_name=req.controller, do_plots=False)
    return {"site_baseline_import": res["site_baseline_import"], "site_control_import": res["site_control_import"]}


