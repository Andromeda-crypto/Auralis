import joblib
import numpy as np
from Nodes.base_node import EnergyNode
from simulation import EnergyWorld, Controller

class AIHybridController(Controller):
    def __init__(self, nn_path: str, rf_path: str):
        self.nn_model = joblib.load(nn_path)
        self.rf_model = joblib.load(rf_path)
        self.history = []

    def select_action(self, node: EnergyNode, world: EnergyWorld, net_flow: float, step: int):
        soc = node.energy / node.capacity
        temp = getattr(node, "temperature", 25)
        eff = node.efficiency
        degr = node.degradation

        features = np.array([[soc, temp, eff, degr, world.energy_supply, world.energy_demand, world.price, net_flow]])
        nn_pred = float(self.nn_model.predict_proba(features)[0][1])
        rf_pred = float(self.rf_model.predict_proba(features)[0][1])
        confidence = 0.5 * (nn_pred + rf_pred)
        base_rate = 10.0
        modulated_rate = base_rate * (0.5 + confidence)

        if net_flow > 0.2:
            return ("charge", modulated_rate)
        elif net_flow < -0.2:
            return ("discharge", modulated_rate)
        else:
            return ("rest", 0.0)
