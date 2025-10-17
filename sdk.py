from typing import Any, Dict

from simulation import (
    HeuristicController,
    MPCLiteController,
    RuleBasedController,
    run_multi_node,
    run_sim,
)

CONTROLLERS = {
    "heuristic": HeuristicController,
    "rule_based": RuleBasedController,
    "mpc_lite": MPCLiteController,
}


def suggest_setpoint(state: Dict[str, Any], controller_name: str = "rule_based") -> Dict[str, Any]:
    # Simple stateless suggestion using controller defaults
    ctrl_cls = CONTROLLERS.get(controller_name, RuleBasedController)
    ctrl = ctrl_cls()
    # Make a lightweight fake world view from state
    class _World:
        def __init__(self, price):
            self.price = price
    class _Node:
        def __init__(self, energy, capacity):
            self.energy = energy
            self.capacity = capacity
    node = _Node(state.get("energy", 50.0), state.get("capacity", 100.0))
    world = _World(state.get("price", 0.20))
    net_flow = state.get("supply", 1.0) - state.get("demand", 1.0)
    action, amount = ctrl.select_action(node, world, net_flow, step=0)
    return {"action": action, "amount": float(amount)}


def run_controller(
    controller_name: str = "rule_based", steps: int = 60, seed: int = 123, do_plots: bool = False
) -> dict:
    ctrl_cls = CONTROLLERS.get(controller_name, RuleBasedController)
    ctrl = ctrl_cls()
    return run_sim(ctrl, steps=steps, seed=seed, do_plots=do_plots)


def run_multinode(
    num_nodes: int = 3,
    feeder_limit: float = 0.8,
    steps: int = 60,
    seed: int = 2024,
    controller_name: str = "rule_based",
    do_plots: bool = False,
):
    ctrl_cls = CONTROLLERS.get(controller_name, RuleBasedController)
    return run_multi_node(num_nodes=num_nodes, feeder_limit=feeder_limit,
                           steps=steps, seed=seed, controller_factory=ctrl_cls, do_plots=do_plots)


