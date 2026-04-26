import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Type
from sumo_rl import SumoEnvironment
from sumo_rl.environment.observations import ObservationFunction

os.environ.setdefault("SUMO_HOME", "D:\\SUMO")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))


@dataclass
class EnvironmentConfig:
    """All parameters needed to build one SumoEnvironment instance."""
    net_file            : str
    route_file          : str
    out_csv_name        : str
    duration            : int   = 3600
    delta_time          : int   = 5
    yellow_time         : int   = 2
    min_green           : int   = 5
    max_green           : int   = 60
    use_gui             : bool  = False
    sumo_warnings       : bool  = False
    additional_sumo_cmd : list  = field(default_factory=list)

    def validate(self):
        assert Path(self.net_file).exists(),   f"net_file not found:   {self.net_file}"
        assert Path(self.route_file).exists(), f"route_file not found: {self.route_file}"

    def additional_sumo_cmd_str(self) -> str:
        """
        sumo-rl calls .split() on additional_sumo_cmd internally so it must
        be a plain space-separated string, not a list.

        Converts: ["--additional-files", "path/to/file.xml"]
              to: "--additional-files path/to/file.xml"

        Note: file paths must not contain spaces. If they do, resolve the
        absolute path and ensure no spaces exist in the directory names.
        """
        return " ".join(self.additional_sumo_cmd)


def make_env(
    config           : EnvironmentConfig,
    reward_fn        : Callable,
    observation_class: Type[ObservationFunction],
    monitor          : bool = True,
    monitor_dir      : Optional[str] = None,
):
    """
    Factory that builds a single SumoEnvironment.
    All paths are forced to str so SUMO never receives a Path object.

    Parameters
    ----------
    config            : EnvironmentConfig — all env parameters
    reward_fn         : callable          — reward function
    observation_class : ObservationFunction subclass
    monitor           : wrap in SB3 Monitor for episode logging
    monitor_dir       : where Monitor writes its CSV (defaults to out_csv_name dir)

    Returns
    -------
    _init : callable — pass directly to DummyVecEnv([make_env(...)])
    """
    config.validate()

    def _init():
        env = SumoEnvironment(
            net_file            = str(config.net_file),
            route_file          = str(config.route_file),
            out_csv_name        = str(config.out_csv_name),
            single_agent        = True,
            use_gui             = config.use_gui,
            num_seconds         = config.duration,
            delta_time          = config.delta_time,
            yellow_time         = config.yellow_time,
            min_green           = config.min_green,
            max_green           = config.max_green,
            reward_fn           = reward_fn,
            observation_class   = observation_class,
            sumo_warnings       = config.sumo_warnings,
            additional_sumo_cmd = config.additional_sumo_cmd_str(),  # str ✓
        )
        if monitor:
            from stable_baselines3.common.monitor import Monitor
            log_dir = monitor_dir or str(Path(config.out_csv_name).parent)
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            return Monitor(env, log_dir)
        return env

    return _init


def make_env_pair(
    train_config     : EnvironmentConfig,
    eval_config      : EnvironmentConfig,
    reward_fn        : Callable,
    observation_class: Type[ObservationFunction],
    train_monitor_dir: Optional[str] = None,
    eval_monitor_dir : Optional[str] = None,
):
    """
    Returns (train_init_fn, eval_init_fn) ready to pass to DummyVecEnv.

    eval_config can use a different route_file or vehicle config to test
    generalisation of the trained agent on unseen traffic scenarios.

    Example
    -------
    train_fn, eval_fn = make_env_pair(train_cfg, eval_cfg, reward_fn, obs_class)
    train_env = DummyVecEnv([train_fn])
    eval_env  = DummyVecEnv([eval_fn])
    """
    train_fn = make_env(
        train_config, reward_fn, observation_class,
        monitor=True, monitor_dir=train_monitor_dir
    )
    eval_fn = make_env(
        eval_config, reward_fn, observation_class,
        monitor=True, monitor_dir=eval_monitor_dir
    )
    return train_fn, eval_fn