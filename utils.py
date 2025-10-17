"""
Plot to visualize and analyze the simulation of nodes, to specify direction and tasks.
"""

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


def plot_metric(history: List[Dict[str, Any]], metric_name: str) -> None:
    values = [entry[metric_name] for entry in history]
    if metric_name == "stability":
        plt.axhline(y=40, color="r", linestyle="--", label="Instability Threshold")
        plt.fill_between(range(len(values)), 0, 40, color="red", alpha=0.1, label="Unstable Zone!")
        plt.legend()
    if metric_name == "energy":
        min_e = min(v for v in values) if not history else history[0].get("min_energy", 0)
        max_e = max(v for v in values) if not history else history[0].get("max_energy", 100)
        plt.axhline(y=min_e, color="gray", linestyle="--", alpha=0.6, label="Min Energy")
        plt.axhline(y=max_e, color="gray", linestyle="--", alpha=0.6, label="Max Energy")
    if metric_name == "temperature":
        cap = history[0].get("max_temperature", 85) if history else 85
        plt.axhline(y=cap, color="orange", linestyle="--", alpha=0.8, label="Temp Cap")
    plt.plot(values, label=metric_name)
    plt.xlabel("Timestep")
    plt.ylabel(metric_name.capitalize())
    plt.title(f"{metric_name.capitalize()} Over Time")
    plt.legend()
    plt.show()


def summarize_performance(history: List[Dict[str, float]]) -> None:
    if not history:
        print("No history to summarize.")
        return
    avg_eff = sum(entry["efficiency"] for entry in history) / len(history)
    avg_deg = sum(entry["degradation"] for entry in history) / len(history)
    avg_stability = sum(entry["stability"] for entry in history) / len(history)
    print(f"Average Efficiency: {avg_eff:.4f}")
    print(f"Average Degradation: {avg_deg:.4f}")
    print(f"Average Stability: {avg_stability:.4f}")


def compute_kpis(
    history: List[Dict[str, Any]],
    stability_threshold: float = 40,
    energy_bounds: Tuple[float, float] = (0, 100),
    temperature_cap: float = 85,
) -> Dict[str, float]:
    """
    Compute basic KPIs over a run history.

    Returns a dict with:
    - mean_efficiency, mean_degradation, mean_stability
    - time_unstable (count below stability threshold)
    - constraint_violations (count of energy/temperature excursions)
    - energy_throughput (sum of absolute energy deltas)
    """
    if not history:
        return {}

    min_e, max_e = energy_bounds
    mean_efficiency = sum(h["efficiency"] for h in history) / len(history)
    mean_degradation = sum(h["degradation"] for h in history) / len(history)
    mean_stability = sum(h["stability"] for h in history) / len(history)

    time_unstable = sum(1 for h in history if h["stability"] < stability_threshold)

    energy_values = [h["energy"] for h in history]
    energy_throughput = sum(
        abs(energy_values[i] - energy_values[i - 1]) for i in range(1, len(energy_values))
    )

    last = history[-1]
    energy_violations = last.get("energy_violations")
    temperature_violations = last.get("temperature_violations")

    if energy_violations is None or temperature_violations is None:
        energy_violations = 0
        temperature_violations = 0
        for h in history:
            if h["energy"] < min_e or h["energy"] > max_e:
                energy_violations += 1
            if h.get("temperature", 0) > temperature_cap:
                temperature_violations += 1

    return {
        "mean_efficiency": mean_efficiency,
        "mean_degradation": mean_degradation,
        "mean_stability": mean_stability,
        "time_unstable": float(time_unstable),
        "energy_violations": float(energy_violations),
        "temperature_violations": float(temperature_violations),
        "constraint_violations": float(energy_violations + temperature_violations),
        "energy_throughput": energy_throughput,
    }


def print_run_report(history: List[Dict[str, Any]], kpis: Dict[str, float]) -> None:
    if not history:
        print("No run data.")
        return
    print("\n=== Run Report ===")
    print(f"Steps: {len(history)}")
    print(f"Mean Efficiency: {kpis.get('mean_efficiency', float('nan')):.4f}")
    print(f"Mean Degradation: {kpis.get('mean_degradation', float('nan')):.4f}")
    print(f"Mean Stability: {kpis.get('mean_stability', float('nan')):.4f}")
    print(f"Time Unstable (<40): {kpis.get('time_unstable', 0):.0f}")
    print(f"Constraint Violations (total): {kpis.get('constraint_violations', 0):.0f}")
    print(f" - Energy Bounds Violations: {kpis.get('energy_violations', 0):.0f}")
    print(f" - Temperature Cap Violations: {kpis.get('temperature_violations', 0):.0f}")
    print(f"Energy Throughput: {kpis.get('energy_throughput', 0):.2f}")


def plot_series(series: List[float], name: str) -> None:
    if not series:
        print(f"No data for {name}.")
        return
    plt.plot(series, label=name)
    plt.xlabel("Timestep")
    plt.ylabel(name.capitalize())
    plt.title(f"{name.capitalize()} Over Time")
    plt.legend()
    plt.show()


def compute_economic_kpis(
    price_series: List[float],
    baseline_import_series: List[float],
    control_import_series: List[float],
) -> Dict[str, float]:
    if not price_series or not baseline_import_series or not control_import_series:
        return {}
    n = min(len(price_series), len(baseline_import_series), len(control_import_series))
    price = price_series[:n]
    base = baseline_import_series[:n]
    ctrl = control_import_series[:n]
    baseline_cost = sum(price[i] * base[i] for i in range(n))
    control_cost = sum(price[i] * ctrl[i] for i in range(n))
    savings = baseline_cost - control_cost
    return {
        "baseline_cost": baseline_cost,
        "control_cost": control_cost,
        "savings": savings,
    }


def print_economic_report(econ_kpis: Dict[str, float]) -> None:
    if not econ_kpis:
        print("No economic data.")
        return
    print("\n=== Economic Report ===")
    print(f"Baseline Cost: ${econ_kpis.get('baseline_cost', 0):.2f}")
    print(f"Control Cost:  ${econ_kpis.get('control_cost', 0):.2f}")
    print(f"Savings:       ${econ_kpis.get('savings', 0):.2f}")


# --- Artifact utilities --- #

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def default_run_id(prefix: str = "run") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def save_history_csv(history: List[Dict[str, Any]], out_path: str) -> None:
    if not history:
        return
    keys = list(history[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def save_json(data: Any, out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def save_plot(fig: Any, out_path: str) -> None:
    fig.savefig(out_path, bbox_inches="tight")

