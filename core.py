from dataclasses import dataclass
from typing import List, Any


class EnergyNode:
    def __init__(self, capacity: float = 100.0):
        self.capacity = capacity
        self.energy = 0.5 * capacity
        self.efficiency = 1.0
        self.degradation = 0.0
        self.temperature = 25.0
        self.voltage = 1.0


class Controller:
    def select_action(self, node: EnergyNode, world: Any, net_flow: float, step: int):
        raise NotImplementedError
