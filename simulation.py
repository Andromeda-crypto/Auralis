from Nodes.base_node import EnergyNode
from utils import plot_metric
from adaptive import adapt_node
import random
import time

class EnergyWorld:
    def __init__(self):
        self.energy_supply = random.uniform(0.5, 1.5)
        self.energy_demand = random.uniform(0.3, 1.2)
        self.noise = random.uniform(-0.05, 0.05)

    def update_conditions(self):
        # small random fluctuations every step
        self.energy_supply = max(0.1, min(self.energy_supply + random.uniform(-0.2, 0.2), 2.0))
        self.energy_demand = max(0.1, min(self.energy_demand + random.uniform(-0.2, 0.2), 2.0))


# --- Run Simulation --- #

node = EnergyNode()
world = EnergyWorld()

for step in range(50):
    world.update_conditions()

    # create realistic nonlinear fluctuations
    world.energy_supply += random.uniform(-0.3, 0.3)
    world.energy_demand += random.uniform(-0.3, 0.3)
    world.energy_supply = max(0.1, min(world.energy_supply, 2.0))
    world.energy_demand = max(0.1, min(world.energy_demand, 2.0))

    # Compute energy imbalance
    net_flow = world.energy_supply - world.energy_demand

    # Non-linear response to imbalance
    if net_flow > 0.2:
        node.charge(net_flow * random.uniform(5, 15))
    elif net_flow < -0.2:
        node.discharge(abs(net_flow) * random.uniform(5, 15))
    else:
        node.rest()

    # Always apply physics & degradation feedback each step
    node.update_physics()
    node.log_state()
    adapt_node(node)

    print(
        f"{step:02d} : Energy = {node.energy:.2f}, "
        f"Eff = {node.efficiency:.3f}, "
        f"Degr = {node.degradation:.3f}, "
        f"Temp = {getattr(node, 'temperature', 25):.2f}, "
        f"State = {node.state}"
    )

    time.sleep(0.05)

# --- Plot after simulation --- #
plot_metric(node.history, "energy")
plot_metric(node.history, "efficiency")
plot_metric(node.history, "degradation")
plot_metric(node.history, "stability")

print("\nSimulation complete.")
node.recall_instability_events()
