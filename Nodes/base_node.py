"""
Creates a Singular Node that behaves like an appliance and has similar characteristics
"""

import random
from typing import Any


class EnergyNode:
    def __init__(self) -> None:
        self.energy: float = 100
        self.degradation: float = 0.0  # zero deg initially
        self.efficiency: float = 1.0  # maximal efficiency to begin with
        self.max_energy: float = 100
        self.min_energy: float = 0
        self.capacity: float = 100
        self.temperature: float = 25.0  # initialize at room temp
        self.stability_threshold: float = 40
        self.state: str = "idle"  # can also be charging or discharging
        self.history: list[dict[str, Any]] = []  # stores past actions and consequences
        self.memory: list[dict[str, Any]] = [] # keeps track of unwanted results and pitfalls 
        self.max_temperature: float = 85.0
        self.energy_violations: int = 0
        self.temperature_violations: int = 0

    # Adding basic "behaviuor of the node"

    def charge(self, amount: float) -> None:
        self.energy = min(self.max_energy, self.energy + amount * self.efficiency)
        self.degradation += 0.001
        self.state = "charging"

    def discharge(self, amount: float) -> None:
        self.energy = max(self.min_energy, self.energy - amount / self.efficiency)
        self.state = "discharging"

    def rest(self) -> None:
        self.state = "idle"

    def update_physics(self) -> None:
        leakage = 0.05 * (1 - self.efficiency) + random.uniform(-0.02, 0.02)
        self.energy -= float(max(0, leakage))

        if self.state == "charging":
            self.temperature += random.uniform(0.2, 0.5)
        elif self.state == "discharging":
            self.temperature += random.uniform(0.3, 0.6)
        else:
            self.temperature -= random.uniform(0.1, 0.3)
        if self.temperature > self.max_temperature:
            self.temperature_violations += 1
            self.temperature = self.max_temperature
            self.degradation += 0.002
        self.temperature = max(20.0, self.temperature)

        # Efficiency dynamically affected by degradation, temp, and energy load
        thermal_penalty = (self.temperature - 25) * 0.0018
        degradation_penalty = self.degradation * random.uniform(0.4, 0.6)
        self.efficiency = max(0.5, 1.0 - degradation_penalty - thermal_penalty)

        stress_factor = 1.0
        if self.temperature > 60:
            stress_factor += (self.temperature - 60) / 20
        if self.energy > 95:
            stress_factor += 0.5
        if self.energy < 10:
            stress_factor += 0.5
        wear = (0.002 if self.state == "idle" else 0.005) * stress_factor
        self.degradation += wear
        self.energy += random.uniform(-0.3, 0.3)

        if self.energy < self.min_energy or self.energy > self.max_energy:
            self.energy_violations += 1
            self.degradation += 0.001  # mild penalty
        self.energy = max(self.min_energy, min(self.energy, self.max_energy))

    """
    Add a logging state to give node a memory of what it has done
    """

    def log_state(self) -> None:
        stability = (self.energy * self.efficiency) / (1 + self.degradation)

        # Log current state
        entry: dict[str, Any] = {
            "energy": self.energy,
            "efficiency": self.efficiency,
            "degradation": self.degradation,
            "stability": stability,
            "temperature": self.temperature,
            "state": self.state,
            # snapshot of cumulative violations to enable per-step deltas if desired
            "energy_violations": self.energy_violations,
            "temperature_violations": self.temperature_violations,
            # bounds for plotting
            "min_energy": self.min_energy,
            "max_energy": self.max_energy,
            "max_temperature": self.max_temperature,
        }
        self.history.append(entry)

        # --- Detect and remember low stability safely ---
        if stability < 40:
            snapshot: dict[str, Any] = {
                "step": len(self.history),
                "energy": self.energy,
                "efficiency": self.efficiency,
                "degradation": self.degradation,
                "state": self.state,
                "stability": stability,
                "temperature": self.temperature,
            }
            # Mark whether this snapshot is below the node's configured threshold
            snapshot["is_unstable"] = snapshot["stability"] < self.stability_threshold
            self.memory.append(snapshot)

    def recall_instability_events(self) -> None:
        if not self.memory:
            print("No instability events recorded")
            return
        print("\n -------Instability Events -------")
        for m in self.memory:
            print(
                f"Step {m['step']}: Stability={m['stability']:.2f}, "
                f"Energy={m['energy']:.2f}, Eff={m['efficiency']:.2f}, "
                f"Degr={m['degradation']:.3f}, State={m['state']}"
            )