"""
Creates a Singular Node that behaves like an appliance and has similar characteristics 
"""
import random 



class EnergyNode:
    def __init__(self):
        self.energy = 100
        self.degradation = 0.  #zero deg initially
        self.efficiency = 1.0  #maximal efficiency to begin with
        self.max_energy = 100
        self.min_energy = 0
        self.state= 'idle' # can also be charging or discharging
        self.history = [] # stores past actions and consequences


   # Adding basic "behaviuor of the node"

    def charge(self, amount):
        self.energy = min(self.max_energy, self.energy + amount*self.efficiency)
        self.degradation += 0.001
        self.state = "charging"

    def discharge(self, amount):
        self.energy = max(self.min_energy, self.energy - amount/self.efficiency)
        self.state = "discharging"

    def rest(self):
        self.state = "idle"

    """
    Add a logging state to give node a memory of what it has done
    """

    def log_state(self):
        self.history.append({
            "energy" : self.energy,
            "efficiency": self.efficiency,
            "degradation": self.degradation,
            "state": self.state
        })

    


        
            

    



    
