from Nodes.base_node import EnergyNode
from simulation import HeuristicController, run_sim


def test_node_initial_state():
    n = EnergyNode()
    assert 0 <= n.energy <= n.capacity
    assert n.efficiency <= 1.0


def test_run_sim_returns_kpis():
    res = run_sim(HeuristicController(), steps=5, seed=1, do_plots=False)
    assert "kpis" in res and "econ" in res
    assert len(res["history"]) == 5
