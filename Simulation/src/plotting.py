import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY = True
except ImportError:
    PLOTLY = False
    print("Plotly not installed — interactive plots disabled.")

# ── Matplotlib / Seaborn ───────────────────────────────────────────────────────

def plot_episode_rewards(episode_csv: str, title: str = "Episode rewards"):
    df  = pd.read_csv(episode_csv)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["episode"], df["total_reward"], lw=1.5, color="#185FA5")
    window = max(1, len(df) // 10)
    ax.plot(df["episode"],
            df["total_reward"].rolling(window).mean(),
            lw=2, color="#D85A30", label=f"Rolling mean ({window} ep)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig


def plot_correlation_heatmap(step_csv: str, variables: list = None):
    df = pd.read_csv(step_csv)
    if variables:
        df = df[variables]
    corr = df.corr().round(2)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                linewidths=0.5, square=True,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation matrix")
    plt.tight_layout()
    plt.show()
    return fig


def plot_causal_variables(step_csv: str, episode: int = 0):
    """Time-series of all causal variables for one episode."""
    df  = pd.read_csv(step_csv)
    ep  = df[df["episode"] == episode]
    vars_ = ["mean_speed", "queue_length", "pressure",
             "waiting_time", "collision_count", "reward"]
    fig, axes = plt.subplots(len(vars_), 1, figsize=(12, 14), sharex=True)
    colors = ["#185FA5","#D85A30","#0F6E56","#BA7517","#A32D2D","#534AB7"]
    for ax, var, col in zip(axes, vars_, colors):
        ax.plot(ep["step"], ep[var], color=col, lw=1.2)
        ax.set_ylabel(var.replace("_", " "), fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Step")
    fig.suptitle(f"Episode {episode} — causal variable traces", fontsize=13)
    plt.tight_layout()
    plt.show()
    return fig


def plot_algorithm_comparison(csv_paths: dict, metric: str = "total_reward"):
    """
    Compare multiple algorithms from their episode CSVs.
    csv_paths = {"PPO": "path/to/ppo_episodes.csv", "A2C": ...}
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    colors  = ["#185FA5", "#D85A30", "#0F6E56", "#534AB7"]
    for (algo, path), col in zip(csv_paths.items(), colors):
        df  = pd.read_csv(path)
        w   = max(1, len(df) // 10)
        ax.plot(df["episode"], df[metric].rolling(w).mean(),
                label=algo, color=col, lw=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"Algorithm comparison — {metric}")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig


def plot_vehicle_config_comparison(csv_paths: dict, metric: str = "total_collisions"):
    """
    Compare outcomes across different vehicle configs (traffic scenarios).
    csv_paths = {"standard": path, "aggressive": path, "slippery": path}
    """
    means = {k: pd.read_csv(v)[metric].mean() for k, v in csv_paths.items()}
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(means.keys(), means.values(),
                  color=["#85B7EB","#F0997B","#97C459","#FAC775"])
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{metric} by vehicle configuration")
    plt.tight_layout()
    plt.show()
    return fig


# ── Plotly interactive ─────────────────────────────────────────────────────────

def plotly_training_dashboard(step_csv: str, episode_csv: str):
    """Interactive dashboard — reward curve + causal variable time-series."""
    if not PLOTLY:
        print("Plotly not available.")
        return

    ep_df   = pd.read_csv(episode_csv)
    step_df = pd.read_csv(step_csv)

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Episode total reward", "Collisions per episode",
            "Mean speed over time", "Queue length over time",
            "Waiting time over time", "Pressure over time",
        ]
    )

    fig.add_trace(go.Scatter(x=ep_df["episode"], y=ep_df["total_reward"],
                             mode="lines", name="reward",
                             line=dict(color="#185FA5")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ep_df["episode"], y=ep_df["total_collisions"],
                             mode="lines", name="collisions",
                             line=dict(color="#A32D2D")), row=1, col=2)
    fig.add_trace(go.Scatter(x=step_df["step"], y=step_df["mean_speed"],
                             mode="lines", name="speed",
                             line=dict(color="#0F6E56", width=0.8)), row=2, col=1)
    fig.add_trace(go.Scatter(x=step_df["step"], y=step_df["queue_length"],
                             mode="lines", name="queue",
                             line=dict(color="#D85A30", width=0.8)), row=2, col=2)
    fig.add_trace(go.Scatter(x=step_df["step"], y=step_df["waiting_time"],
                             mode="lines", name="waiting",
                             line=dict(color="#BA7517", width=0.8)), row=3, col=1)
    fig.add_trace(go.Scatter(x=step_df["step"], y=step_df["pressure"],
                             mode="lines", name="pressure",
                             line=dict(color="#534AB7", width=0.8)), row=3, col=2)

    fig.update_layout(height=800, title_text="PPO Training Dashboard",
                      showlegend=False)
    fig.show()
    return fig


def plotly_scatter_outcomes(step_csv: str,
                            x: str = "queue_length",
                            y: str = "waiting_time",
                            color: str = "collision_count"):
    if not PLOTLY:
        return
    df  = pd.read_csv(step_csv)
    fig = px.scatter(df, x=x, y=y, color=color,
                     color_continuous_scale="RdYlGn_r",
                     title=f"{y} vs {x} coloured by {color}",
                     opacity=0.6)
    fig.show()
    return fig