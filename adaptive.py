# adaptive.py
"""
Adaptive Learning Module for EnergyNode
Reads node memory, detects instability patterns, and applies corrective adjustments.
Designed to be modular and extensible for future metrics and specialized nodes.
"""

from Nodes.base_node import EnergyNode


def adapt_node(node: EnergyNode, lookback: int = 20, stability_threshold: float = 40.0) -> None:
    """
    Perform adaptive corrections on the node based on past instability events.

    Parameters:
    - node: EnergyNode instance
    - lookback: number of past memory entries to consider
    - stability_threshold: threshold below which a state is considered unstable
    """

    # Extract recent unstable events
    recent_instabilities = [
        m for m in node.memory[-lookback:] if m.get("stability", 100) < stability_threshold
    ]

    if not recent_instabilities:
        return  # nothing to adapt

    # Calculate average conditions during instability
    _avg_energy = sum(m["energy"] for m in recent_instabilities) / len(recent_instabilities)
    avg_temp = sum(m["temperature"] for m in recent_instabilities) / len(recent_instabilities)
    avg_eff = sum(m["efficiency"] for m in recent_instabilities) / len(recent_instabilities)

    # Example adaptive strategies
    # 1️⃣ Reduce energy slightly if node was frequently unstable at high temp
    if node.temperature > avg_temp:
        node.energy *= 0.97

    # 2️⃣ Boost efficiency slightly if node often degraded under certain conditions
    if node.efficiency < avg_eff:
        node.efficiency *= 1.01

    # 3️⃣ Optional: adjust degradation slightly to simulate preventive maintenance
    node.degradation *= 0.995

    # 4️⃣ Placeholder for future metrics (voltage, current, vibrations)
