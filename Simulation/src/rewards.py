import numpy as np
from sumo_rl.environment.traffic_signal import TrafficSignal


def _get_waiting_time(ts: TrafficSignal) -> float:
    raw = ts.get_accumulated_waiting_time_per_lane()
    if isinstance(raw, dict):   return float(sum(raw.values()))
    if isinstance(raw, (int, float)): return float(raw)
    return float(sum(raw))

def _get_pressure(ts: TrafficSignal) -> float:
    raw = ts.get_pressure()
    if isinstance(raw, list): return float(sum(raw))
    if isinstance(raw, dict): return float(sum(raw.values()))
    return float(raw)

def _get_queue(ts: TrafficSignal) -> float:
    return float(sum(
        ts.sumo.lane.getLastStepHaltingNumber(l) for l in ts.lanes
    ))

def _get_collisions(ts: TrafficSignal) -> float:
    return float(len(ts.sumo.simulation.getCollisions()))


# ── Reward functions — each accepts a TrafficSignal, returns float ─────────────

def weighted_reward(ts: TrafficSignal) -> float:
    """
    Normalized reward in [-1, 0]. Weights sum to 1.0.
    Best general-purpose reward for single-intersection training.
    """
    MAX = dict(waiting=300.0, queue=20.0, pressure=10.0, collision=1.0)

    norm_w = min(_get_waiting_time(ts) / MAX["waiting"],  1.0)
    norm_q = min(_get_queue(ts)        / MAX["queue"],    1.0)
    norm_p = min(abs(_get_pressure(ts))/ MAX["pressure"], 1.0)
    norm_c = min(_get_collisions(ts)   / MAX["collision"],1.0)

    return -(norm_w * 0.3 + norm_q * 0.3 + norm_p * 0.2 + norm_c * 0.2)


def queue_reward(ts: TrafficSignal) -> float:
    """Penalise only queue length. Simple baseline."""
    return -_get_queue(ts)


def waiting_time_reward(ts: TrafficSignal) -> float:
    """Penalise only accumulated waiting time."""
    return -_get_waiting_time(ts)


def pressure_reward(ts: TrafficSignal) -> float:
    """Penalise only lane imbalance (pressure). Fast learning signal."""
    return -abs(_get_pressure(ts))


def safety_weighted_reward(ts: TrafficSignal) -> float:
    """
    Higher collision weight (0.4) for safety-critical scenarios.
    Useful when testing slippery / aggressive vehicle configs.
    """
    MAX = dict(waiting=300.0, queue=20.0, pressure=10.0, collision=1.0)

    norm_w = min(_get_waiting_time(ts) / MAX["waiting"],  1.0)
    norm_q = min(_get_queue(ts)        / MAX["queue"],    1.0)
    norm_p = min(abs(_get_pressure(ts))/ MAX["pressure"], 1.0)
    norm_c = min(_get_collisions(ts)   / MAX["collision"],1.0)

    return -(norm_w * 0.2 + norm_q * 0.2 + norm_p * 0.2 + norm_c * 0.4)


# Registry — lets agent.py / notebooks select reward by name
REWARD_REGISTRY = {
    "weighted"       : weighted_reward,
    "queue"          : queue_reward,
    "waiting_time"   : waiting_time_reward,
    "pressure"       : pressure_reward,
    "safety_weighted": safety_weighted_reward,
}


def get_reward_fn(name: str):
    assert name in REWARD_REGISTRY, \
        f"Unknown reward '{name}'. Choose from {list(REWARD_REGISTRY.keys())}"
    return REWARD_REGISTRY[name]