import math
import random
import statistics
import time
import os
import csv
from dataclasses import dataclass
from typing import List, Dict, Any

from core import EnergyNode, Controller
from ai_controller import AIHybridController


@dataclass
class EnergyWorld:
    energy_demand: float
    energy_supply: float
    price: float
    avg_temp: float
    timestep: int = 0

    def update(self):
        """Simulate dynamic environmental & market conditions each step."""
        self.timestep += 1
        self.energy_demand = max(0, self.energy_demand * (1 + random.uniform(-0.05, 0.05)))
        self.energy_supply = max(0, self.energy_supply * (1 + random.uniform(-0.05, 0.05)))
        self.price = max(0.01, self.price * (1 + random.uniform(-0.02, 0.02)))
        self.avg_temp += random.uniform(-0.3, 0.3)


class HeuristicController(Controller):
    def select_action(self, node, world, net_flow, step):
        if net_flow > 0.3:
            return ("charge", 8.0)
        elif net_flow < -0.3:
            return ("discharge", 8.0)
        else:
            return ("rest", 0.0)

class RuleBasedController(Controller):
    def select_action(self, node, world, net_flow, step):
        if node.energy < 0.3 * node.capacity:
            return ("charge", 10.0)
        elif node.energy > 0.8 * node.capacity:
            return ("discharge", 10.0)
        else:
            return ("rest", 0.0)

class MPCLiteController(Controller):
    def select_action(self, node, world, net_flow, step):
        trend = math.sin(step / 20.0)
        if trend > 0 and net_flow > 0:
            return ("charge", 9.0)
        elif trend < 0 and net_flow < 0:
            return ("discharge", 9.0)
        else:
            return ("rest", 0.0)

class CvxpyMPCController(Controller):
    def select_action(self, node, world, net_flow, step):
        if world.price < 0.5:
            return ("charge", 10.0)
        elif world.price > 1.5:
            return ("discharge", 10.0)
        else:
            return ("rest", 0.0)

def simulate(world: EnergyWorld, controller: Controller, steps: int = 200) -> Dict[str, Any]:
    node = EnergyNode()
    history = []
    total_savings = 0.0
    total_violations = 0
    total_temp_deviation = 0.0

    for step in range(steps):
        net_flow = world.energy_supply - world.energy_demand
        action, rate = controller.select_action(node, world, net_flow, step)

        if action == "charge":
            node.energy = min(node.capacity, node.energy + rate * 0.1)
            total_savings += rate * (1.0 - world.price)
        elif action == "discharge":
            node.energy = max(0.0, node.energy - rate * 0.1)
            total_savings += rate * world.price * 0.8

        node.temperature += (world.avg_temp - node.temperature) * 0.05
        node.efficiency *= 1 + random.uniform(-0.01, 0.01)
        node.voltage = 1.0 + random.uniform(-0.05, 0.05)

        if node.energy > node.capacity or node.energy < 0:
            total_violations += 1

        total_temp_deviation += abs(node.temperature - world.avg_temp)
        world.update()

        history.append({
            "step": step,
            "energy": node.energy,
            "price": world.price,
            "action": action,
            "rate": rate,
            "temp": node.temperature,
            "avg_temp": world.avg_temp,
            "savings": total_savings,
        })

    return {
        "controller": controller.__class__.__name__,
        "total_savings": total_savings,
        "violations": total_violations,
        "mean_efficiency": statistics.mean(([h["rate"] for h in history if h["rate"] > 0]) if history else 0.0),
        "mean_temp_deviation": total_temp_deviation / steps,
        "steps": steps
    }


def log_to_csv(result: Dict[str, Any], filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

def run_leaderboard(runs: int = 3, steps: int = 200):
    controllers: List[Controller] = [
        HeuristicController(),
        RuleBasedController(),
        MPCLiteController(),
        CvxpyMPCController(),
        AIHybridController(
            nn_path="models/neural_instability_predictor.pkl",
            rf_path="models/random_forest_instability.pkl"
        ),
    ]

    results = []
    csv_path = "results/leaderboard_log.csv"

    for controller in controllers:
        all_runs = []
        for _ in range(runs):
            world = EnergyWorld(
                energy_demand=100.0,
                energy_supply=120.0,
                price=1.0,
                avg_temp=25.0
            )
            result = simulate(world, controller, steps)
            all_runs.append(result)
            log_to_csv(result, csv_path)

        avg_savings = statistics.mean([r["total_savings"] for r in all_runs])
        avg_violations = statistics.mean([r["violations"] for r in all_runs])
        avg_temp_dev = statistics.mean([r["mean_temp_deviation"] for r in all_runs])
        results.append((controller.__class__.__name__, avg_savings, avg_violations, avg_temp_dev))

    print("\nLeaderboard (mean over runs)")
    print("{:<25} {:>12} {:>15} {:>18}".format("Controller", "Savings", "Violations", "Temp Deviation"))
    for name, savings, viol, temp_dev in results:
        print(f"{name:<25} ${savings:>10.2f} {viol:>14.2f} {temp_dev:>18.3f}")

if __name__ == "__main__":
    start = time.time()
    run_leaderboard(runs=3, steps=300)
    print(f"\nSimulation completed in {time.time() - start:.2f}s.")
