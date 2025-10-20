import math
import statistics
from typing import Dict, Any

#  Helper Functions

def _safe_value(val: float) -> float:
    """Clamp NaN and inf values to 0 to avoid corrupting learning."""
    if math.isnan(val) or math.isinf(val):
        return 0.0
    return val


def _calculate_reinforcement_score(pre_state: Dict[str, float], post_state: Dict[str, float]) -> float:
    """
    Calculate reinforcement score based on relative improvements in energy, voltage, and efficiency.
    Positive = improvement, Negative = degradation.
    """
    weights = {"energy": 0.4, "voltage": 0.3, "efficiency": 0.3}
    score = 0.0

    for key, weight in weights.items():
        pre_val = _safe_value(pre_state.get(key, 0.0))
        post_val = _safe_value(post_state.get(key, 0.0))
        delta = post_val - pre_val
        norm = max(abs(pre_val), 1e-6)
        score += weight * (delta / norm)

    return score


def adjust_learning_rate(node: Any, window: int = 10) -> float:
    """
    Adjust learning rate based on average reinforcement score in recent history.
    """
    log = getattr(node, "reinforcement_log", [])
    if not log:
        return getattr(node, "learning_rate", 1.0)

    recent_scores = [entry["score"] for entry in log[-window:]]
    avg_score = statistics.mean(recent_scores)
    base_learning = getattr(node, "learning_rate", 1.0)

    if avg_score > 0.01:
        new_lr = min(base_learning * (1 + avg_score), 1.2)
    elif avg_score < -0.01:
        new_lr = max(base_learning * (1 + avg_score), 0.8)
    else:
        new_lr = base_learning

    return round(new_lr, 4)


def _trim_logs(node: Any, max_log: int = 500) -> None:
    """Prevent logs from growing indefinitely."""
    if len(node.reinforcement_log) > max_log:
        node.reinforcement_log.pop(0)
    if len(node.history) > max_log:
        node.history.pop(0)



# Core Adaptive Node Function


def adapt_node(node: Any, network_metrics: Dict[str, float]) -> None:
    """
    Adjust the node's parameters based on network-wide averages and its own performance.
    Incorporates reinforcement learning score and dynamic learning rate.
    """
    avg_temp = network_metrics.get("avg_temp", 0.0)
    avg_voltage = network_metrics.get("avg_voltage", 1.0)

    # Initialize adaptive attributes if not present
    if not hasattr(node, "reinforcement_log"):
        node.reinforcement_log = []
    if not hasattr(node, "learning_rate"):
        node.learning_rate = 1.0
    if not hasattr(node, "history"):
        node.history = []

    # Record pre-adaptation state
    pre_state = {
        "energy": _safe_value(node.energy),
        "voltage": _safe_value(node.voltage),
        "efficiency": _safe_value(node.efficiency),
    }

    # Use learning rate directly in this cycle’s corrections
    lr = node.learning_rate

    # Adaptive corrections based on deviation from network state
    if node.temperature > avg_temp + 2.0:
        node.energy *= 0.95 * lr
        node.efficiency *= 1.02 * lr

    if avg_voltage < 0.95 or avg_voltage > 1.05:
        node.energy *= 0.98 * lr
        node.voltage = (node.voltage + 1.0) / 2

    # Stability damping to prevent runaway oscillations
    node.energy = max(node.energy, 0.0)
    node.voltage = max(min(node.voltage, 1.2), 0.8)
    node.efficiency = max(min(node.efficiency, 1.5), 0.0)

    # Record post-adaptation state
    post_state = {
        "energy": _safe_value(node.energy),
        "voltage": _safe_value(node.voltage),
        "efficiency": _safe_value(node.efficiency),
    }

    # Calculate and record reinforcement score
    reinforcement_score = _calculate_reinforcement_score(pre_state, post_state)
    node.reinforcement_log.append({"score": reinforcement_score})

    
    node.learning_rate = adjust_learning_rate(node, window=10)

    # Maintain history for analysis/debugging
    node.history.append({
        "pre": pre_state,
        "post": post_state,
        "reinforcement": reinforcement_score,
        "learning_rate": node.learning_rate
    })

    # Trim logs to prevent memory issues on long runs
    _trim_logs(node, max_log=500)
