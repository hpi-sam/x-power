from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)

ALGORITHMS = {"PPO": PPO, "A2C": A2C, "DQN": DQN}

# ── Default hyperparameters per algorithm ──────────────────────────────────────
ALGO_DEFAULTS = {
    "PPO": dict(
        learning_rate = 3e-4,
        n_steps       = 512,
        batch_size    = 64,
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        ent_coef      = 0.01,
    ),
    "A2C": dict(
        learning_rate = 7e-4,
        n_steps       = 5,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        ent_coef      = 0.01,
    ),
    "DQN": dict(
        learning_rate       = 1e-4,
        buffer_size         = 10_000,
        learning_starts     = 1_000,
        batch_size          = 64,
        gamma               = 0.99,
        exploration_fraction= 0.1,
        exploration_final_eps=0.05,
    ),
}


@dataclass
class AgentConfig:
    """All parameters for building and training an RL agent."""
    algorithm        : str  = "PPO"
    total_timesteps  : int  = 50_000
    n_eval_episodes  : int  = 3
    eval_freq        : int  = 5_000
    checkpoint_freq  : int  = 5_000
    model_dir        : str  = "outputs/models/PPO"
    log_dir          : str  = "outputs/logs/PPO"
    hyperparameters  : dict = field(default_factory=dict)  # overrides defaults

    def __post_init__(self):
        assert self.algorithm in ALGORITHMS, \
            f"Unknown algorithm '{self.algorithm}'. Choose from {list(ALGORITHMS.keys())}"
        # Merge defaults with any user overrides
        defaults = ALGO_DEFAULTS[self.algorithm].copy()
        defaults.update(self.hyperparameters)
        self.hyperparameters = defaults
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)


class AgentBuilder:
    """
    Builds, trains, and evaluates an SB3 agent from an AgentConfig.

    Usage
    -----
    builder = AgentBuilder(config)
    model   = builder.build(train_env)
    model   = builder.train(model, train_env, eval_env, extra_callbacks=[])
    results = builder.evaluate(model, eval_env)
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    def build(self, train_env: DummyVecEnv):
        """Instantiate the algorithm with config hyperparameters."""
        AlgoClass = ALGORITHMS[self.config.algorithm]
        model = AlgoClass(
            policy        = "MlpPolicy",
            env           = train_env,
            tensorboard_log = self.config.log_dir,
            verbose       = 1,
            **self.config.hyperparameters,
        )
        print(f"✓ Built {self.config.algorithm} | "
              f"hyperparams: {self.config.hyperparameters}")
        return model

    def build_callbacks(
        self,
        eval_env        : DummyVecEnv,
        extra_callbacks : list = [],
    ) -> list:
        """Standard callbacks + any extras (e.g. CausalLoggingCallback)."""
        return [
            *extra_callbacks,
            CheckpointCallback(
                save_freq   = self.config.checkpoint_freq,
                save_path   = self.config.model_dir,
                name_prefix = self.config.algorithm,
                verbose     = 1,
            ),
            EvalCallback(
                eval_env,
                best_model_save_path = self.config.model_dir,
                log_path             = self.config.log_dir,
                eval_freq            = self.config.eval_freq,
                n_eval_episodes      = self.config.n_eval_episodes,
                deterministic        = True,
                verbose              = 1,
            ),
        ]

    def train(self, model, train_env, eval_env,
              extra_callbacks: list = []):
        """Train the model and save the final checkpoint."""
        callbacks = self.build_callbacks(eval_env, extra_callbacks)
        model.learn(
            total_timesteps = self.config.total_timesteps,
            callback        = callbacks,
            tb_log_name     = self.config.algorithm,
        )
        final_path = str(Path(self.config.model_dir) /
                         f"{self.config.algorithm}_final")
        model.save(final_path)
        print(f"\n✓ Training complete → {final_path}")
        return model

    def evaluate(self, model, eval_env_fn, n_episodes: Optional[int] = None):
        """
        Run deterministic rollouts and return per-episode reward list.
        eval_env_fn — the _init callable (not yet constructed), so each
        episode gets a fresh env.
        """
        import numpy as np
        n = n_episodes or self.config.n_eval_episodes
        raw_env     = eval_env_fn()
        ep_rewards  = []

        for ep in range(n):
            obs, _       = raw_env.reset()
            total_reward = 0.0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = raw_env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            ep_rewards.append(total_reward)
            print(f"  Eval episode {ep+1:02d} | reward: {total_reward:.3f}")

        raw_env.close()
        print(f"\n  Mean: {np.mean(ep_rewards):.3f} | "
              f"Std: {np.std(ep_rewards):.3f}")
        return ep_rewards

    @staticmethod
    def load(algorithm: str, model_path: str):
        """Load a saved model from disk."""
        model = ALGORITHMS[algorithm].load(model_path)
        print(f"✓ Loaded {algorithm} from {model_path}")
        return model