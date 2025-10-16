from Nodes.base_node import EnergyNode
import random
import time

"""
Create a world in which the node lives and operates
"""

class EnergyWorld:
    def __init__(self):
        self.energy_supply = random.uniform(0.5,1.5) # power available
        self.energy_demand = random.uniform(0.3,1.2) # power required
        self.noise = random.uniform(-0.05,0.05)  # Some noise to make it unpredictable

    def update_conditions(self):
        self.energy_supply = max(0, self.energy_supply + random.uniform(-0.1,0.1))
        self.energy_demand = max(0, self.energy_demand + random.uniform(-0.1, 0.1))


# --- Run Simulation --- #

node = EnergyNode()
world = EnergyWorld()

for step in range(50):
    world.update_conditions()
    if world.energy_supply > world.energy_demand :
        node.charge(world.energy_supply - world.energy_demand)
    else:
        node.discharge(world.energy_demand - world.energy_supply)

    node.log_state()
    print(f"{step} : Energy = {node.energy:.2f}, Efficiency = {node.efficiency}, Degradation = {node.degradation}, State = {node.state}")
    time.sleep(0.1)
