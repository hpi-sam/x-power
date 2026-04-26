import csv
import numpy as np
from pathlib import Path
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback
from sumo_rl import SumoEnvironment
from sumo_rl.environment.traffic_signal import TrafficSignal


class StepLogger:
    """
    Logs per-step causal variables to CSV.
    Can be used:
      (a) during agent training via CausalLoggingCallback
      (b) standalone during env-only trace collection
    """

    FIELDS = [
        "episode", "step", "reward",
        "phase_action",
        "mean_speed", "total_speed", "queue_length",
        "pressure", "vehicle_count", "density",
        "waiting_time", "collision_count",
    ]

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.buffer   = []
        self.episode  = 0
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()
        print(f"✓ StepLogger initialised → {csv_path}")

    @staticmethod
    def _extract(ts: TrafficSignal, action: int, reward: float,
                 episode: int, step: int) -> dict:
        """Pull all variables from a TrafficSignal object into a flat dict."""

        # Pressure
        p = ts.get_pressure()
        pressure = float(sum(p) if isinstance(p, list)
                         else sum(p.values()) if isinstance(p, dict)
                         else p)

        # Waiting time
        w = ts.get_accumulated_waiting_time_per_lane()
        waiting = float(sum(w.values()) if isinstance(w, dict)
                        else w if isinstance(w, (int, float))
                        else sum(w))

        lane_speeds = [ts.sumo.lane.getLastStepMeanSpeed(l) for l in ts.lanes]
        lane_counts = [ts.sumo.lane.getLastStepVehicleNumber(l) for l in ts.lanes]
        lane_caps   = [ts.sumo.lane.getLength(l) / 7.5 for l in ts.lanes]

        return {
            "episode"        : episode,
            "step"           : step,
            "reward"         : round(float(reward), 6),
            "phase_action"   : int(action),
            "mean_speed"     : round(float(np.mean(lane_speeds)), 4),
            "total_speed"    : round(float(sum(lane_speeds)),     4),
            "queue_length"   : round(float(sum(
                ts.sumo.lane.getLastStepHaltingNumber(l) for l in ts.lanes
            )), 2),
            "pressure"       : round(pressure,  4),
            "vehicle_count"  : round(float(sum(lane_counts)), 0),
            "density"        : round(float(np.mean([
                c / max(cap, 1e-6)
                for c, cap in zip(lane_counts, lane_caps)
            ])), 4),
            "waiting_time"   : round(waiting, 2),
            "collision_count": int(len(ts.sumo.simulation.getCollisions())),
        }

    def log(self, env: SumoEnvironment, action: int,
            reward: float, step: int):
        ts = env.traffic_signals[env.ts_ids[0]]
        row = self._extract(ts, action, reward, self.episode, step)
        self.buffer.append(row)

    def flush(self):
        """Write buffer to CSV. Call at episode end."""
        if not self.buffer:
            return
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerows(self.buffer)
        self.buffer  = []
        self.episode += 1

    def load(self):
        """Load the full CSV into a pandas DataFrame."""
        import pandas as pd
        return pd.read_csv(self.csv_path)


class EpisodeSummaryLogger:
    """Logs one row per episode — total reward, mean metrics."""

    FIELDS = [
        "episode", "total_reward", "steps",
        "mean_speed", "mean_queue",
        "mean_waiting", "total_collisions",
    ]

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.episode  = 0
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def log_episode(self, step_buffer: list):
        if not step_buffer:
            return
        import numpy as np
        row = {
            "episode"          : self.episode,
            "total_reward"     : round(sum(r["reward"]          for r in step_buffer), 3),
            "steps"            : len(step_buffer),
            "mean_speed"       : round(float(np.mean([r["mean_speed"]     for r in step_buffer])), 4),
            "mean_queue"       : round(float(np.mean([r["queue_length"]   for r in step_buffer])), 2),
            "mean_waiting"     : round(float(np.mean([r["waiting_time"]   for r in step_buffer])), 2),
            "total_collisions" : sum(r["collision_count"] for r in step_buffer),
        }
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(row)
        self.episode += 1
        print(f"  Episode {self.episode:03d} | "
              f"reward: {row['total_reward']:7.3f} | "
              f"collisions: {row['total_collisions']} | "
              f"mean_speed: {row['mean_speed']:.3f}")

    def load(self):
        import pandas as pd
        return pd.read_csv(self.csv_path)


class CausalLoggingCallback(BaseCallback):
    """SB3 callback — logs causal variables every training step."""

    def __init__(self, step_logger: StepLogger,
                 episode_logger: Optional[EpisodeSummaryLogger] = None,
                 verbose: int = 0):
        super().__init__(verbose)
        self.step_logger    = step_logger
        self.episode_logger = episode_logger
        self.step_count     = 0

    def _get_raw_env(self) -> SumoEnvironment:
        return self.training_env.envs[0].env   # DummyVecEnv→Monitor→SumoEnv

    def _on_step(self) -> bool:
        env    = self._get_raw_env()
        action = self.locals["actions"][0]
        reward = self.locals["rewards"][0]

        self.step_logger.log(env, action, reward, self.step_count)
        self.step_count += 1

        if self.locals["dones"][0]:
            if self.episode_logger:
                self.episode_logger.log_episode(self.step_logger.buffer)
            self.step_logger.flush()

        return True