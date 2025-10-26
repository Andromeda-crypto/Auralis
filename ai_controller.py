import joblib
import numpy as np
from core import EnergyNode, Controller
from typing import Any  # Use Any to avoid circular import

class AIHybridController(Controller):
    def __init__(self, nn_path: str, rf_path: str):
        super().__init__()
        self.nn_model = joblib.load(nn_path)
        self.rf_model = joblib.load(rf_path)
        self.history = []

    def select_action(self, node: EnergyNode, world: Any, net_flow: float, step: int):
        soc = node.energy / node.capacity
        temp = getattr(node, "temperature", 25)
        eff = node.efficiency
        degr = node.degradation

        features = np.array([[soc, temp, eff, degr, getattr(world, 'energy_supply', 0),
                              getattr(world, 'energy_demand', 0), getattr(world, 'price', 0),
                              net_flow]], dtype=np.float32)

        try:
            nn_pred = float(self.nn_model.predict_proba(features)[0][1])
            rf_pred = float(self.rf_model.predict_proba(features)[0][1])
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return ("rest", 0.0)

        confidence = 0.5 * (nn_pred + rf_pred)
        base_rate = 10.0
        modulated_rate = base_rate * (0.5 + confidence)

        if net_flow > 0.2:
            action = ("charge", modulated_rate)
        elif net_flow < -0.2:
            action = ("discharge", modulated_rate)
        else:
            action = ("rest", 0.0)

        self.history.append({
            "step": step,
            "soc": soc,
            "net_flow": net_flow,
            "confidence": confidence,
            "action": action[0]
        })

        return action
