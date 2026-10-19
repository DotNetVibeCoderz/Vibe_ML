"""Figures for the forecast report."""
import matplotlib.pyplot as plt


def plot_forecast(actual, predicted, path="forecast.png"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(actual)), actual, label="actual", color="0.4")
    ax.plot(range(len(predicted)), predicted, label="forest", color="tab:blue")
    ax.set_title("Held-out days: actual vs predicted sales")
    ax.set_xlabel("day")
    ax.set_ylabel("units")
    ax.legend()
    fig.savefig(path)
    return path


def plot_importances(pairs, path="importances.png"):
    names = [name for name, _ in pairs]
    values = [value for _, value in pairs]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(range(len(names)), values, color="tab:orange")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_title("Feature importance")
    fig.tight_layout()
    fig.savefig(path)
    return path
