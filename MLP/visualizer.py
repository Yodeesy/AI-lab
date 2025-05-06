# ===== visualizer.py =====
import matplotlib.pyplot as plt

def plot_single_curve(losses, optimizer_name, metric_name="Loss"):
    plt.figure()
    plt.plot(losses, label=f"{optimizer_name} {metric_name}")
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f"Training {metric_name} - {optimizer_name}")
    plt.savefig(f"{metric_name.lower()}_curve_{optimizer_name}.png", dpi=300)
    plt.close()

def plot_compare_curve(losses_sgd, losses_adam, metric_name="Loss"):
    plt.figure()
    plt.plot(losses_sgd, label=f"SGD {metric_name}")
    plt.plot(losses_adam, label=f"Adam {metric_name}")
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f"Training {metric_name} Comparison")
    plt.savefig(f"{metric_name.lower()}_curve_compare.png", dpi=300)
    plt.close()
