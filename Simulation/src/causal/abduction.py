import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from src.causal.scm import TrafficSCM


class AbductionTree:
    """
    Structural Abduction: given an observed outcome (e.g. high reward or collision),
    trace back through the causal graph to find the most probable root causes.

    Abduction steps:
      1. Observe outcome  (e.g. reward=low)
      2. Infer latent     (work backwards through SCM)
      3. Rank causes      (by posterior probability)
      4. Visualise tree   (highlight causal path)
    """

    def __init__(self, scm: TrafficSCM):
        self.scm   = scm
        self.graph = scm.graph

    def abduce(self, outcome_var: str, outcome_val: str,
               top_k: int = 3) -> list[dict]:
        """
        Given outcome_var=outcome_val, find the top_k most probable causes.

        Returns list of dicts sorted by posterior probability.
        """
        # Get all ancestor nodes of the outcome
        ancestors = nx.ancestors(self.graph, outcome_var)

        results = []
        for cause_var in ancestors:
            for cause_val in ["low", "medium", "high", "0", "1"]:
                try:
                    posterior = self.scm.query(
                        target=outcome_var,
                        evidence={cause_var: cause_val}
                    )
                    # Extract P(outcome_val | cause_var=cause_val)
                    prob = dict(zip(
                        posterior.state_names[outcome_var],
                        posterior.values
                    )).get(outcome_val, 0.0)

                    results.append({
                        "cause_variable" : cause_var,
                        "cause_value"    : cause_val,
                        "outcome"        : f"{outcome_var}={outcome_val}",
                        "probability"    : round(float(prob), 4),
                    })
                except Exception:
                    continue

        # Sort by highest posterior
        results = sorted(results, key=lambda x: x["probability"], reverse=True)
        return results[:top_k]

    def plot_abduction_tree(self, outcome_var: str, outcome_val: str,
                            top_k: int = 3, save_path: str = None):
        """
        Visualise the abduction tree: outcome node + top causal paths highlighted.
        """
        top_causes = self.abduce(outcome_var, outcome_val, top_k)

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # ── Left: full causal graph with highlighted paths ────────────────────
        ax1  = axes[0]
        pos  = nx.spring_layout(self.graph, seed=42)
        
        highlight_nodes = {outcome_var} | {c["cause_variable"] for c in top_causes}
        node_colors = [
            "red"       if n == outcome_var        else
            "orange"    if n in highlight_nodes    else
            "lightblue"
            for n in self.graph.nodes
        ]

        nx.draw(
            self.graph, pos, ax=ax1,
            with_labels=True,
            node_color=node_colors,
            node_size=2000,
            arrowsize=15,
            font_size=9,
            font_weight="bold"
        )
        ax1.set_title(f"Causal Graph\n(red=outcome, orange=top causes)")

        # ── Right: abduction tree as ranked bar chart ─────────────────────────
        ax2    = axes[1]
        labels = [f"{c['cause_variable']}\n={c['cause_value']}" for c in top_causes]
        probs  = [c["probability"] for c in top_causes]
        colors = ["#e74c3c", "#e67e22", "#f1c40f"][:len(top_causes)]

        bars = ax2.barh(labels, probs, color=colors, edgecolor="black")
        ax2.set_xlabel("P(outcome | cause)")
        ax2.set_title(
            f"Abduction Tree\nTop {top_k} causes of {outcome_var}={outcome_val}"
        )
        ax2.set_xlim(0, 1)

        for bar, prob in zip(bars, probs):
            ax2.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{prob:.3f}", va="center", fontsize=10
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

        return top_causes