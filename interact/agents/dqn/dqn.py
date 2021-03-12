from typing import Tuple, Dict, List, Callable, Optional

import gin
import gym
import tensorflow as tf

from interact.agents.base import Agent
from interact.agents.dqn.policy import DQNPolicy
from interact.agents.utils import register
from interact.experience.runner import Runner
from interact.experience.sample_batch import SampleBatch
from interact.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from interact.schedules import LinearDecay
from interact.typing import TensorType


@gin.configurable(name_or_fn="dqn", denylist=["env_fn"])
@register("dqn")
class DQNAgent(Agent):
    """The Deep Q-Network algorithm.

    Args:
        env_fn: A function that, when called, returns an instance of the agent's
            environment.
        q_network: The type of model to use for the policy network.
        batch_size: The size of experience batches sampled from the replay buffer.
        buffer_size: The size of the experience replay buffer. Network update are
            sampled from this number of most recent frames.
        train_freq: The frequency (measured in the number of updates) with which the
            online Q-network is updated.
        target_update_freq: The frequency (measured in the number of updates) with
            which the target network is updated.
        gamma: Discount factor.
        lr: The learning rate.
        initial_epsilon: Initial value of epsilon in e-greedy exploration.
        final_epsilon: Final value of epsilon in e-greedy exploration.
        epsilon_timesteps: The number of timesteps over which epsilon is annealed.
        learning_starts: A uniform random policy is run for this number of steps before
            learning starts, and the resulting experience is used to populate the
            replay memory.
        max_grad_norm: The maximum value for the gradient clipping.
        double: If True, Double Q-Learning is used.
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
    """

    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        q_network: str = "mlp",
        batch_size: int = 32,
        buffer_size: int = 50000,
        train_freq: int = 1,
        target_update_freq: int = 500,
        gamma: float = 0.99,
        lr: float = 5e-4,
        initial_epsilon: float = 1.0,
        final_epsilon: float = 0.02,
        epsilon_timesteps: int = 10000,
        learning_starts: int = 1000,
        max_grad_norm: Optional[float] = None,
        double: bool = True,
        prioritized_replay: bool = False,
        prioritized_replay_alpha: float = 0.6,
        prioritized_replay_beta: float = 0.4,
        final_prioritized_replay_beta: float = 4.0,
        prioritized_replay_beta_steps: Optional[int] = None,
        prioritized_replay_epsilon: float = 1e-6,
    ):
        super().__init__(env_fn)

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self.gamma = gamma
        self.lr = lr
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_timesteps = epsilon_timesteps
        self.learning_starts = learning_starts
        self.max_grad_norm = max_grad_norm
        self.double = double
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta = prioritized_replay_beta
        self.prioritized_replay_beta_steps = prioritized_replay_beta_steps
        self.final_prioritized_replay_beta = final_prioritized_replay_beta
        self.prioritized_replay_epsilon = prioritized_replay_epsilon

        env = self.make_env()
        self.policy = DQNPolicy(env.observation_space, env.action_space, q_network)

        self.optimizer = None
        self.epsilon = None
        self.replay_buffer = None
        self.beta_schedule = None

        # Because we are only using one local worker, we can just provide a reference
        # to the learner's policy and not worry about updating the actor's weights
        # during training. This speeds things up considerably.
        self.runner_config = dict(
            env_fn=env_fn,
            policy_fn=lambda: self.policy,
            num_envs_per_worker=1,
            num_workers=1,
        )
        self.runner = None

    @property
    def timesteps_per_iteration(self) -> int:
        return 1

    @tf.function
    def _update(self, obs, actions, rewards, next_obs, dones, weights):
        actions = tf.cast(actions, tf.int32)

        # Get Q values for the observations at the next time step with the
        # target network
        next_q_values = self.policy.target_network(next_obs)

        if self.double:
            # If using double DQN, we use online network to select actions
            selected_actions = tf.argmax(self.policy(next_obs), axis=-1)
            # Extract only the Q-values for our selected actions
            next_q_values_best = tf.gather(
                next_q_values, selected_actions, batch_dims=1
            )
        else:
            # In vanilla DQN, simply use the maximum value
            next_q_values_best = tf.reduce_max(next_q_values, axis=-1)

        # Set Q values for terminal experiences to zero
        next_q_values_best_masked = (1.0 - dones) * next_q_values_best
        # Calculate targets (Bellman equation)
        targets = rewards + self.gamma * next_q_values_best_masked
        targets = tf.expand_dims(targets, 1)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.policy.q_network.trainable_variables)

            # Forward pass through online Q network
            q_values = self.policy(obs)
            q_values_selected = tf.gather(q_values, actions, batch_dims=1)
            q_values_selected = tf.expand_dims(q_values_selected, 1)

            # Huber loss implements the gradient clipping described by Mnih et. al.
            loss = tf.losses.huber(targets, q_values_selected)
            loss = tf.reduce_mean(loss * weights)

        # Calculate gradients of the loss with respect to the network parameters
        vars = self.policy.q_network.trainable_variables
        grads = tape.gradient(loss, vars)
        # Perform gradient clipping
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        # Apply the gradient update
        self.optimizer.apply_gradients(zip(grads, vars))

        td_errors = q_values_selected - targets

        return {"loss": loss}, td_errors

    @tf.function
    def act(self, obs: TensorType, deterministic: bool = True) -> TensorType:
        q_values = self.policy(obs)

        if deterministic:
            actions = tf.argmax(q_values, axis=-1)
        else:
            raise ValueError("Non-deterministic actions are not implemented for DQN.")

        return actions

    def train(self, update: int) -> Tuple[Dict[str, float], List[Dict]]:
        # Get the current value of epsilon for exploration
        current_epsilon = self.epsilon(update)
        # Take a step in the environment
        episodes, ep_infos = self.runner.run(1, epsilon=current_epsilon)

        # Add the new experience to the replay buffer
        self.replay_buffer.add(episodes.to_sample_batch())

        metrics = {"epsilon": current_epsilon}

        if update > self.learning_starts and update % self.train_freq == 0:
            # Sample a batch of experience from the replay buffer
            if self.prioritized_replay:
                sample = self.replay_buffer.sample(
                    self.batch_size, self.beta_schedule(update)
                )
                weights = sample[SampleBatch.PRIO_WEIGHTS]
            else:
                sample = self.replay_buffer.sample(self.batch_size)
                weights = 1.0

            # Perform an update step using the sampled experience
            metrics_from_update, td_errors = self._update(
                sample[SampleBatch.OBS],
                sample[SampleBatch.ACTIONS],
                sample[SampleBatch.REWARDS],
                sample[SampleBatch.NEXT_OBS],
                sample[SampleBatch.DONES],
                weights,
            )

            metrics.update(metrics_from_update)

            if self.prioritized_replay:
                self.replay_buffer.update_priorities(sample["batch_indices"], td_errors)

        # Periodically update target network
        if update > self.learning_starts and update % self.target_update_freq == 0:
            self.policy.update_target_network()

        return metrics, ep_infos

    def pretrain_setup(self, total_timesteps: int):
        self.epsilon = LinearDecay(
            initial_learning_rate=self.initial_epsilon,
            decay_steps=self.epsilon_timesteps,
            end_learning_rate=self.final_epsilon,
        )

        self.beta_schedule = LinearDecay(
            initial_learning_rate=self.prioritized_replay_beta,
            decay_steps=self.prioritized_replay_beta_steps or total_timesteps,
            end_learning_rate=self.final_prioritized_replay_beta,
        )

        self.runner = Runner(**self.runner_config)
        self.optimizer = tf.optimizers.Adam(learning_rate=self.lr)

        if self.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.buffer_size, self.prioritized_replay_alpha
            )
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size)
