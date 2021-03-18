from typing import Callable, Optional

import gin
import gym

from interact.agents.ddpg.ddpg import DDPGAgent
from interact.agents.utils import register


@gin.configurable(name_or_fn="td3", denylist=["env_fn"])
@register("td3")
class TD3Agent(DDPGAgent):
    """The Twin Delayed DDPG (TD3) algorithm.

    This algorithm is a minor modification of DDPG. This class is merely a wrapper
    around DDPG with the TD3 features enabled by default. Namely, TD3 uses twin
    critic networks, delayed policy updates, and target policy smoothing.

    Args:
        env_fn: A function that, when called, returns an instance of the agent's
            environment.
        network: Base network type to be used by the policy and Q-functions.
        actor_lr: Learning rate to use for updating the actor.
        critic_lr: Learning rate to use for updating the critics.
        tau: Parameter for the polyak averaging used to update the target networks.
        target_update_interval: Frequency with which the target Q-networks are updated.
        gamma: The discount factor.
        buffer_size: The maximum size of the replay buffer.
        train_freq: The frequency with which training updates are performed.
        target_update_interval: The frequency with which the target network is updated.
        learning_starts: The number of timesteps after which learning starts.
        random_steps: Actions will be sampled completely at random for this many
            timesteps at the beginning of training.
        batch_size: The size of batches sampled from the replay buffer over which
            updates are performed.
        num_workers: The number of parallel workers to use for experience collection.
        num_envs_per_worker: The number of synchronous environments to be executed in
            each worker.
        prioritized_replay: If True, a prioritized experience replay will be used.
        prioritized_replay_alpha: Alpha parameter for prioritized replay.
        prioritized_replay_beta: Initial beta parameter for prioritized replay.
        final_prioritized_replay_beta: The final value of the prioritized replay beta
            parameter.
        prioritized_replay_beta_steps: Number of steps over which the prioritized
            replay beta parameter will be annealed. If None, this will be set to the
            total number of training steps.
        prioritized_replay_epsilon: Epsilon to add to td-errors when updating
            priorities.
        initial_noise_scale: The initial scale of the Gaussian noise that is added to
            actions for exploration.
        final_noise_scale: The final scale of the Gaussian noise that is added to
            actions for exploration.
        noise_scale_steps: The number of timesteps over which the amount of exploration
            noise is annealed from `initial_noise_scale` to `final_noise_scale`. If
            None, the total duration of training is used.
        use_huber: If True, the Huber loss is used in favor of MSE for critic updates.
        use_twin_critic: If True, twin critic networks are used.
        policy_delay: The policy is updated once for every `policy_delay` critic
            updates.
        smooth_target_policy: If true, target policy smoothing is used in the critic
            updates.
        target_noise: The amount of target noise that is used for smoothing.
        target_noise_clip: The value at which target noise is clipped.
    """

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
