import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Install: pip install pgmpy


class TrafficSCM:
    """
    Causal graph for traffic signal control:

    time_of_day → vehicle_density
    vehicle_density → queue_length → waiting_time → reward
    vehicle_density → speed        → waiting_time
    phase_action    → queue_length
    phase_action    → speed
    collision       → reward
    """

    CAUSAL_EDGES = [
        ("time_of_day",     "vehicle_density"),
        ("vehicle_density", "queue_length"),
        ("vehicle_density", "speed"),
        ("phase_action",    "queue_length"),
        ("phase_action",    "speed"),
        ("queue_length",    "waiting_time"),
        ("speed",           "waiting_time"),
        ("waiting_time",    "reward"),
        ("collision",       "reward"),
    ]

    def __init__(self):
        self.model     = BayesianNetwork(self.CAUSAL_EDGES)
        self.graph     = nx.DiGraph(self.CAUSAL_EDGES)
        self.inference = None

    def fit(self, data: pd.DataFrame):
        """Fit CPDs from collected episode data."""
        data_disc = self._discretize(data)
        self.model.fit(
            data_disc,
            estimator=MaximumLikelihoodEstimator
        )
        self.inference = VariableElimination(self.model)
        print("✓ SCM fitted")

    def _discretize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Bin continuous variables into low/medium/high."""
        df = data.copy()
        for col in ["vehicle_density", "queue_length",
                    "waiting_time", "speed"]:
            if col in df.columns:
                df[col] = pd.qcut(
                    df[col], q=3,
                    labels=["low", "medium", "high"],
                    duplicates="drop"
                )
        # Binary columns
        for col in ["collision", "phase_action"]:
            if col in df.columns:
                df[col] = df[col].astype(str)
        return df

    def query(self, target: str, evidence: dict):
        """P(target | evidence)"""
        assert self.inference, "Call fit() first"
        result = self.inference.query(
            variables=[target],
            evidence=evidence,
            show_progress=False
        )
        return result

    def plot(self, save_path: str = None):
        plt.figure(figsize=(12, 8))
        pos = {
            "time_of_day"     : (0,  2),
            "vehicle_density" : (1,  2),
            "phase_action"    : (1,  0),
            "queue_length"    : (2,  2),
            "speed"           : (2,  0),
            "waiting_time"    : (3,  1),
            "collision"       : (3, -1),
            "reward"          : (4,  1),
        }
        nx.draw(
            self.graph, pos,
            with_labels=True,
            node_color="lightblue",
            node_size=2500,
            arrowsize=20,
            font_size=10,
            font_weight="bold"
        )
        plt.title("Structural Causal Model — Traffic Signal Control")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()