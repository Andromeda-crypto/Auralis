import random
import time
from typing import Any, Callable, Tuple

from adaptive import adapt_node
from Nodes.base_node import EnergyNode
from utils import (
    compute_economic_kpis,
    compute_kpis,
    default_run_id,
    ensure_dir,
    plot_metric,
    plot_series,
    print_economic_report,
    print_run_report,
    save_history_csv,
    save_json,
    summarize_performance,
)

try:
    import cvxpy as cvxpy_module

    cp: Any = cvxpy_module
except Exception:
    cp = None


class EnergyWorld:
    def __init__(self) -> None:
        self.energy_supply = random.uniform(0.5, 1.5)
        self.energy_demand = random.uniform(0.3, 1.2)
        self.noise = random.uniform(-0.05, 0.05)
        self.price = 0.20

    def update_conditions(self) -> None:
        self.energy_supply = max(
            0.1, min(self.energy_supply + random.uniform(-0.2, 0.2), 2.0)
        )
        self.energy_demand = max(
            0.1, min(self.energy_demand + random.uniform(-0.2, 0.2), 2.0)
        )
        self.price = max(0.05, min(self.price + random.uniform(-0.02, 0.02), 0.50))


# --- Controllers --- #
class Controller:
    def name(self) -> str:
        return self.__class__.__name__

    def select_action(
        self, node: EnergyNode, world: EnergyWorld, net_flow: float, step: int
    ) -> Tuple[str, float]:
        return ("rest", 0.0)


class HeuristicController(Controller):
    def select_action(
        self, node: EnergyNode, world: EnergyWorld, net_flow: float, step: int
    ) -> Tuple[str, float]:
        if net_flow > 0.2:
            return ("charge", net_flow * random.uniform(5, 15))
        if net_flow < -0.2:
            return ("discharge", abs(net_flow) * random.uniform(5, 15))
        return ("rest", 0.0)


class RuleBasedController(Controller):
    def __init__(
        self,
        low_price: float = 0.12,
        high_price: float = 0.30,
        low_soc: float = 30.0,
        high_soc: float = 80.0,
    ) -> None:
        self.low_price = low_price
        self.high_price = high_price
        self.low_soc = low_soc
        self.high_soc = high_soc

    def select_action(
        self, node: EnergyNode, world: EnergyWorld, net_flow: float, step: int
    ) -> Tuple[str, float]:
        soc = 100.0 * (node.energy / node.capacity)
        if world.price <= self.low_price and soc < self.high_soc:
            magnitude = max(0.0, (self.high_soc - soc) / 100.0) * 12.0
            return ("charge", magnitude)
        if world.price >= self.high_price and soc > self.low_soc:
            magnitude = max(0.0, (soc - self.low_soc) / 100.0) * 12.0
            return ("discharge", magnitude)
        return ("rest", 0.0)


class MPCLiteController(Controller):
    def __init__(self, horizon: int = 12) -> None:
        self.horizon = horizon
        self.window: list[float] = []

    def select_action(
        self, node: EnergyNode, world: EnergyWorld, net_flow: float, step: int
    ) -> Tuple[str, float]:
        self.window.append(world.price)
        if len(self.window) > 24:
            self.window.pop(0)
        if not self.window:
            return ("rest", 0.0)
        sorted_w = sorted(self.window)
        low_q = sorted_w[max(0, int(0.25 * len(sorted_w)) - 1)]
        high_q = sorted_w[min(len(sorted_w) - 1, int(0.75 * len(sorted_w)))]
        soc = 100.0 * (node.energy / node.capacity)
        if world.price <= low_q and soc < 85.0:
            return ("charge", 10.0)
        if world.price >= high_q and soc > 35.0:
            return ("discharge", 10.0)
        return ("rest", 0.0)


class CvxpyMPCController(Controller):
    def __init__(
        self,
        horizon: int = 12,
        max_rate: float = 12.0,
        min_soc: float = 0.2,
        max_soc: float = 0.9,
    ) -> None:
        self.horizon = horizon
        self.max_rate = max_rate
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.window: list[float] = []

    def select_action(
        self, node: EnergyNode, world: EnergyWorld, net_flow: float, step: int
    ) -> Tuple[str, float]:
        if cp is None:
            return MPCLiteController(self.horizon).select_action(
                node, world, net_flow, step
            )

        self.window.append(world.price)
        if len(self.window) > 48:
            self.window.pop(0)
        last_price = self.window[-1]
        forecast = [
            self.window[-i - 1] if i < len(self.window) else last_price
            for i in range(self.horizon)
        ]
        forecast = list(reversed(forecast))

        soc0 = max(0.0, min(1.0, node.energy / node.capacity))
        u = cp.Variable(self.horizon)
        soc = cp.Variable(self.horizon + 1)
        constraints = [soc[0] == soc0]
        for t in range(self.horizon):
            constraints += [
                soc[t + 1] == soc[t] + (u[t] / node.capacity),
                cp.abs(u[t]) <= self.max_rate,
            ]
        constraints += [soc >= self.min_soc, soc <= self.max_soc]

        objective = cp.Minimize(
            cp.sum([-forecast[t] * u[t] for t in range(self.horizon)])
            + 0.01 * cp.sum_squares(u)
        )
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS, warm_start=True, verbose=False)
        except Exception:
            return ("rest", 0.0)
        if u.value is None:
            return ("rest", 0.0)
        u0 = float(u.value[0])
        if u0 > 1e-3:
            return ("charge", min(self.max_rate, u0))
        if u0 < -1e-3:
            return ("discharge", min(self.max_rate, abs(u0)))
        return ("rest", 0.0)


def run_sim(
    controller: Controller,
    steps: int = 50,
    seed: int = 42,
    do_plots: bool = False,
    run_id: str | None = None,
    out_dir: str = "artifacts",
) -> dict[str, Any]:
    random.seed(seed)
    node = EnergyNode()
    world = EnergyWorld()
    prices: list[float] = []
    baseline_import: list[float] = []
    control_import: list[float] = []

    for step in range(steps):
        world.update_conditions()
        world.energy_supply += random.uniform(-0.3, 0.3)
        world.energy_demand += random.uniform(-0.3, 0.3)
        world.energy_supply = max(0.1, min(world.energy_supply, 2.0))
        world.energy_demand = max(0.1, min(world.energy_demand, 2.0))

        net_flow = world.energy_supply - world.energy_demand
        prices.append(world.price)

        baseline_import.append(max(0.0, world.energy_demand - world.energy_supply))

        action_type, amount = controller.select_action(node, world, net_flow, step)
        if action_type == "charge":
            node.charge(amount)
            control_import.append(max(0.0, world.energy_demand - world.energy_supply))
        elif action_type == "discharge":
            node.discharge(amount)
            control_import.append(
                max(0.0, world.energy_demand - world.energy_supply)
                - min(amount / 10.0, 1.0)
            )
        else:
            node.rest()
            control_import.append(max(0.0, world.energy_demand - world.energy_supply))

        node.update_physics()
        node.log_state()
        adapt_node(node,network_metrics="history")

        print(
            f"{controller.name()} {step:02d} : Energy = {node.energy:.2f}, "
            f"Eff = {node.efficiency:.3f}, "
            f"Degr = {node.degradation:.3f}, "
            f"Temp = {getattr(node, 'temperature', 25):.2f}, "
            f"State = {node.state}"
        )
        time.sleep(0.02)

    if do_plots:
        plot_metric(node.history, "energy")
        plot_metric(node.history, "efficiency")
        plot_metric(node.history, "degradation")
        plot_metric(node.history, "stability")
        plot_metric(node.history, "temperature")
        plot_series(prices, "price")

    print("\nSimulation complete.")
    node.recall_instability_events()

    kpis = compute_kpis(
        node.history,
        stability_threshold=node.stability_threshold,
        energy_bounds=(node.min_energy, node.max_energy),
        temperature_cap=node.max_temperature,
    )
    print_run_report(node.history, kpis)
    summarize_performance(node.history)

    econ = compute_economic_kpis(prices, baseline_import, control_import)
    print_economic_report(econ)

    # Save artifacts
    run_id = run_id or default_run_id(prefix=f"{controller.name()}")
    out_path = f"{out_dir}/{run_id}"
    ensure_dir(out_path)
    save_history_csv(node.history, f"{out_path}/history.csv")
    save_json(
        {"seed": seed, "controller": controller.name(), "kpis": kpis, "econ": econ},
        f"{out_path}/summary.json",
    )

    return {
        "history": node.history,
        "prices": prices,
        "baseline_import": baseline_import,
        "control_import": control_import,
        "kpis": kpis,
        "econ": econ,
    }


# --- Run all controllers --- #
if __name__ == "__main__":
    controllers: list[Controller] = [
        HeuristicController(),
        RuleBasedController(),
        MPCLiteController(),
        CvxpyMPCController(),
    ]

    results: dict[str, Any] = {}
    seed_base = 123
    for idx, ctrl in enumerate(controllers):
        print(f"\n=== Running controller: {ctrl.name()} ===")
        results[ctrl.name()] = run_sim(
            ctrl, steps=60, seed=seed_base + idx, do_plots=True
        )

    print("\nAll controller runs complete.")


def run_leaderboard(
    ctrls: list[Controller] | None = None,
    steps: int = 60,
    seeds: tuple[int, ...] = (101, 102, 103),
) -> list[dict[str, Any]]:
    ctrls = ctrls or [
        HeuristicController(),
        RuleBasedController(),
        MPCLiteController(),
        CvxpyMPCController(),
    ]
    rows: list[dict[str, Any]] = []
    for ctrl in ctrls:
        savings_list: list[float] = []
        viol_list: list[int] = []
        for s in seeds:
            res = run_sim(ctrl, steps=steps, seed=int(s), do_plots=False)
            savings_list.append(res["econ"].get("savings", 0.0))
            viol_list.append(res["kpis"].get("constraint_violations", 0))
        rows.append(
            {
                "controller": ctrl.name(),
                "mean_savings": sum(savings_list) / len(savings_list),
                "total_violations": sum(viol_list),
            }
        )
    print("\n=== Leaderboard (mean over seeds) ===")
    for r in rows:
        print(
            f"{r['controller']:<20} "
            f"savings=${r['mean_savings']:.2f} "
            f"violations={r['total_violations']}"
        )

    return rows


#  Multi-node with shared feeder constraint
def _default_controller_factory() -> Controller:
    """Default controller factory for multi-node simulation."""
    return RuleBasedController()


def run_multi_node(
    num_nodes: int = 3,
    feeder_limit: float = 0.8,
    steps: int = 60,
    seed: int = 999,
    controller_factory: Callable[[], Controller] | None = None,
    run_id: str | None = None,
    out_dir: str = "artifacts",
    do_plots: bool = True,
) -> dict[str, Any]:
    if controller_factory is None:
        controller_factory = _default_controller_factory

    random.seed(seed)
    world = EnergyWorld()
    nodes: list[EnergyNode] = [EnergyNode() for _ in range(num_nodes)]
    ctrls: list[Controller] = [controller_factory() for _ in range(num_nodes)]

    prices: list[float] = []
    site_baseline_import: list[float] = []
    site_control_import: list[float] = []
    per_node_histories: list[list[dict[str, Any]]] = [[] for _ in range(num_nodes)]

    for step in range(steps):
        world.update_conditions()
        world.energy_supply += random.uniform(-0.3, 0.3)
        world.energy_demand += random.uniform(-0.3, 0.3)
        world.energy_supply = max(0.1, min(world.energy_supply, 2.0))
        world.energy_demand = max(0.1, min(world.energy_demand, 2.0))

        prices.append(world.price)
        baseline_import = max(0.0, world.energy_demand - world.energy_supply)
        site_baseline_import.append(baseline_import)

        required_reduction = max(0.0, baseline_import - feeder_limit)

        actions: list[Tuple[str, float]] = []
        if required_reduction > 0:
            socs = [100.0 * (n.energy / n.capacity) for n in nodes]
            capacities = [max(0.0, (soc - 30.0) / 70.0) for soc in socs]  # 0..1
            total_cap = sum(capacities) or 1.0
            for cap in capacities:
                share = (cap / total_cap) * required_reduction
                amount = min(15.0, share * 10.0)  # cap per-step discharge magnitude
                actions.append(("discharge", amount))
        else:
            for i in range(num_nodes):
                net_flow = world.energy_supply - world.energy_demand
                actions.append(ctrls[i].select_action(nodes[i], world, net_flow, step))

        total_import = baseline_import
        total_offset = 0.0
        for i in range(num_nodes):
            a_type, amt = actions[i]
            if a_type == "charge":
                nodes[i].charge(amt)
            elif a_type == "discharge":
                nodes[i].discharge(amt)
                total_offset += min(amt / 10.0, 1.0)
            else:
                nodes[i].rest()

        total_import = max(0.0, baseline_import - total_offset)
        site_control_import.append(total_import)

        # Physics, logging, adaptation per node
        for i in range(num_nodes):
            nodes[i].update_physics()
            nodes[i].log_state()
            adapt_node(nodes[i])
            per_node_histories[i] = nodes[i].history
        print(
            f"MultiNode step {step:02d}: "
            f"import={total_import:.3f}/{feeder_limit:.3f} "
            f"price={world.price:.2f}"
        )

        time.sleep(0.01)

    print("\nMulti-node simulation complete.")
    for idx, n in enumerate(nodes):
        print(f"\n-- Node {idx} report --")
        k = compute_kpis(
            n.history,
            stability_threshold=n.stability_threshold,
            energy_bounds=(n.min_energy, n.max_energy),
            temperature_cap=n.max_temperature,
        )
        print_run_report(n.history, k)

    econ = compute_economic_kpis(prices, site_baseline_import, site_control_import)
    print("\n-- Site economic report --")
    print_economic_report(econ)

    run_id = run_id or default_run_id(prefix="multinode")
    out_path = f"{out_dir}/{run_id}"
    ensure_dir(out_path)

    save_json(
        {"seed": seed, "num_nodes": num_nodes, "feeder_limit": feeder_limit},
        f"{out_path}/site_meta.json",
    )
    save_json(
        {
            "prices": prices,
            "site_baseline_import": site_baseline_import,
            "site_control_import": site_control_import,
            "econ": econ,
        },
        f"{out_path}/site_series.json",
    )

    for idx, hist in enumerate(per_node_histories):
        save_history_csv(hist, f"{out_path}/node_{idx}_history.csv")

    if do_plots and per_node_histories:
        for metric in ["energy", "efficiency", "degradation", "stability", "temperature"]:
            plot_metric(per_node_histories[0], metric)
        plot_series(prices, "price")

    """# example run can be :
    # --- Demo: Multi-node run ---
    print("\n=== Running multi-node demo (shared feeder limit) ===")
    _multi_results = run_multi_node(
        num_nodes=3, feeder_limit=0.8, steps=60, seed=2024, do_plots=True
    )

    return {
        "nodes": nodes,
        "prices": prices,
        "site_baseline_import": site_baseline_import,
        "site_control_import": site_control_import,
    }"""