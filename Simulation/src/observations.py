import numpy as np
from gymnasium import spaces
from sumo_rl.environment.traffic_signal import TrafficSignal
from sumo_rl.environment.observations import ObservationFunction


class PhaseQueueSpeedObservation(ObservationFunction):
    """
    Default observation:
    [phase_one_hot | density_per_lane | queue_per_lane | speed_per_lane]
    All values normalized to [0, 1].
    """
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        ts       = self.ts
        phase_id = [1 if ts.green_phase == i else 0
                    for i in range(ts.num_green_phases)]
        density  = [
            ts.sumo.lane.getLastStepVehicleNumber(l) /
            (ts.sumo.lane.getLength(l) / 7.5)
            for l in ts.lanes
        ]
        queue = [
            ts.sumo.lane.getLastStepHaltingNumber(l) /
            (ts.sumo.lane.getLength(l) / 7.5)
            for l in ts.lanes
        ]
        speed = [
            ts.sumo.lane.getLastStepMeanSpeed(l) /
            max(ts.sumo.lane.getMaxSpeed(l), 1e-6)
            for l in ts.lanes
        ]
        return np.clip(
            np.array(phase_id + density + queue + speed, dtype=np.float32),
            0.0, 1.0
        )

    def observation_space(self) -> spaces.Box:
        size = self.ts.num_green_phases + 3 * len(self.ts.lanes)
        return spaces.Box(low=0.0, high=1.0, shape=(size,), dtype=np.float32)


class MinimalObservation(ObservationFunction):
    """
    Lightweight observation — phase + queue only.
    Faster convergence, less informative.
    [phase_one_hot | queue_per_lane]
    """
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        ts       = self.ts
        phase_id = [1 if ts.green_phase == i else 0
                    for i in range(ts.num_green_phases)]
        queue    = [
            ts.sumo.lane.getLastStepHaltingNumber(l) /
            (ts.sumo.lane.getLength(l) / 7.5)
            for l in ts.lanes
        ]
        return np.clip(
            np.array(phase_id + queue, dtype=np.float32), 0.0, 1.0
        )

    def observation_space(self) -> spaces.Box:
        size = self.ts.num_green_phases + len(self.ts.lanes)
        return spaces.Box(low=0.0, high=1.0, shape=(size,), dtype=np.float32)


class SSMObservation(ObservationFunction):
    """
    Extended observation that includes SSM-derived safety signals.
    [phase_one_hot | density | queue | speed | mean_ttc_per_lane]
    Requires SSM device enabled in TrafficConfig.
    """
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        ts       = self.ts
        phase_id = [1 if ts.green_phase == i else 0
                    for i in range(ts.num_green_phases)]
        density  = [
            ts.sumo.lane.getLastStepVehicleNumber(l) /
            (ts.sumo.lane.getLength(l) / 7.5)
            for l in ts.lanes
        ]
        queue = [
            ts.sumo.lane.getLastStepHaltingNumber(l) /
            (ts.sumo.lane.getLength(l) / 7.5)
            for l in ts.lanes
        ]
        speed = [
            ts.sumo.lane.getLastStepMeanSpeed(l) /
            max(ts.sumo.lane.getMaxSpeed(l), 1e-6)
            for l in ts.lanes
        ]
        # TTC proxy — mean speed / (queue + 1) per lane as safety signal
        ttc_proxy = [
            (ts.sumo.lane.getLastStepMeanSpeed(l) /
             max(ts.sumo.lane.getLastStepHaltingNumber(l) + 1, 1)) / 10.0
            for l in ts.lanes
        ]
        return np.clip(
            np.array(phase_id + density + queue + speed + ttc_proxy,
                     dtype=np.float32),
            0.0, 1.0
        )

    def observation_space(self) -> spaces.Box:
        size = self.ts.num_green_phases + 4 * len(self.ts.lanes)
        return spaces.Box(low=0.0, high=1.0, shape=(size,), dtype=np.float32)


# Registry
OBSERVATION_REGISTRY = {
    "default" : PhaseQueueSpeedObservation,
    "minimal" : MinimalObservation,
    "ssm"     : SSMObservation,
}

def get_observation_class(name: str):
    assert name in OBSERVATION_REGISTRY, \
        f"Unknown observation '{name}'. Choose from {list(OBSERVATION_REGISTRY.keys())}"
    return OBSERVATION_REGISTRY[name]