"""
Plot to visualize and analyze the simulation of nodes, to specify direction and tasks
"""

import matplotlib.pyplot as plt

def plot_metric(history,metric_name):
    values = [entry[metric_name] for entry in history]
    if metric_name =="stability":
        plt.axhline(y=40,color='r', linestyle = '--', label="Instability Threshold")
        plt.fill_between(
            range(len(values)), 0, 40, color='red', alpha=0.1, label ="Unstable Zone!"
        )
        plt.legend()
    plt.plot(values, label=metric_name)
    plt.xlabel("Timestop")
    plt.ylabel(metric_name.capitalize())
    plt.title(f"{metric_name.capitalize()} Over Time")
    plt.legend()
    plt.show()


def summarize_performance(history):
    avg_eff = sum(entry["efficieccy"] for entry in history / len(history))
    avg_deg = sum(entry["degradation"] for entry in history / len(history))
    print(f"Average Efficiency: {avg_eff:.4f}")
    print(f"Average Degradation: {avg_deg:.4f}")
    

