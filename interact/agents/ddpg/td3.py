from typing import Callable, Optional

import gin
import gym

from interact.agents.ddpg.ddpg import DDPGAgent
from interact.agents.utils import register


@gin.configurable(name_or_fn="td3", denylist=["env_fn"])
@register("td3")
class TD3Agent(DDPGAgent):
    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        network: str = "mlp",
        critic_lr: float = 1e-3,
        actor_lr: float = 1e-3,
        learning_starts: int = 10000,
        random_steps: int = 10000,
        target_update_interval: int = 1,
        tau: float = 0.005,
        gamma: float = 0.95,
        buffer_size: int = 100000,
        train_freq: int = 1,
        batch_size: int = 100,
        num_workers: int = 1,
        num_envs_per_worker: int = 1,
        prioritized_replay: bool = False,
        prioritized_replay_alpha: float = 0.6,
        prioritized_replay_beta: float = 0.4,
        final_prioritized_replay_beta: float = 4.0,
        prioritized_replay_beta_steps: Optional[int] = None,
        prioritized_replay_epsilon: float = 1e-6,
        initial_noise_scale: float = 0.1,
        final_noise_scale: float = 0.1,
        noise_scale_steps: Optional[int] = None,
        use_huber: bool = False,
        use_twin_critic: bool = True,
        policy_delay: int = 2,
        smooth_target_policy: bool = True,
        target_noise: float = 0.2,
        target_noise_clip: float = 0.5,
    ):
        super().__init__(
            env_fn,
            network,
            critic_lr,
            actor_lr,
            learning_starts,
            random_steps,
            target_update_interval,
            tau,
            gamma,
            buffer_size,
            train_freq,
            batch_size,
            num_workers,
            num_envs_per_worker,
            prioritized_replay,
            prioritized_replay_alpha,
            prioritized_replay_beta,
            final_prioritized_replay_beta,
            prioritized_replay_beta_steps,
            prioritized_replay_epsilon,
            initial_noise_scale,
            final_noise_scale,
            noise_scale_steps,
            use_huber,
            use_twin_critic,
            policy_delay,
            smooth_target_policy,
            target_noise,
            target_noise_clip,
        )
