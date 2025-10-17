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
        self.capacity = 100
        self.temperature = 25.0 # initialize at room temp
        self.stability_threshold = 40
        self.state= 'idle' # can also be charging or discharging
        self.history = [] # stores past actions and consequences
        self.memory = []
        # Safety constraints configuration
        self.max_temperature = 85.0
        # Per-run counters for constraint violations
        self.energy_violations = 0
        self.temperature_violations = 0


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

    def update_physics(self):
    # Random natural energy leakage (depends on efficiency + random noise)
        leakage = 0.05 * (1 - self.efficiency) + random.uniform(-0.02, 0.02)
        self.energy -= max(0, leakage)

    # Temperature dynamics (introduce small variance)
        if self.state == "charging":
            self.temperature += random.uniform(0.2, 0.5)
        elif self.state == "discharging":
            self.temperature += random.uniform(0.3, 0.6)
        else:
            self.temperature -= random.uniform(0.1, 0.3)
        # Check and cap temperature; count violation if exceeded
        if self.temperature > self.max_temperature:
            self.temperature_violations += 1
            self.temperature = self.max_temperature
            # penalty: extra wear when overheated
            self.degradation += 0.002
        self.temperature = max(20, self.temperature)

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
    # Slight random drift in energy
        self.energy += random.uniform(-0.3, 0.3)
        # Enforce energy bounds; count violation if exceeded
        if self.energy < self.min_energy or self.energy > self.max_energy:
            self.energy_violations += 1
            self.degradation += 0.001  # mild penalty
        self.energy = max(self.min_energy, min(self.energy, self.max_energy))
                            
        

    
    """
    Add a logging state to give node a memory of what it has done
    """

    def log_state(self):
        stability = (self.energy * self.efficiency) / (1 + self.degradation)

    # Log current state
        entry = {
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
        "max_temperature": self.max_temperature
    }
        self.history.append(entry)

    # --- Detect and remember low stability safely ---
        if stability < 40:  
            snapshot = {
            "step": len(self.history),
            "energy": self.energy,
            "efficiency": self.efficiency,
            "degradation": self.degradation,
            "state": self.state,
            "stability": stability,
            "temperature": self.temperature
        }
            # Mark whether this snapshot is below the node's configured threshold
            snapshot["is_unstable"] = snapshot["stability"] < self.stability_threshold
            self.memory.append(snapshot)

    def recall_instability_events(self):
        if not self.memory :
            print("No instability events recorded")
            return
        print("\n -------Instability Events -------")
        for m in self.memory:
            print(
            f"Step {m['step']}: Stability={m['stability']:.2f}, "
            f"Energy={m['energy']:.2f}, Eff={m['efficiency']:.2f}, "
            f"Degr={m['degradation']:.3f}, State={m['state']}"
        )

    


        
            

    



    
