[1]: https://arxiv.org/abs/1509.02971

[2]: https://arxiv.org/abs/1802.09477

# Deep Deterministic Policy Gradient

[Deep Deterministic Policy Gradient][1] (DDPG) can be thought of as a version of
Q-learning that is compatible with continuous actions. It is an actor-critic method with
a deterministic actor, and it adopts many of the tricks that allowed [DQN](../dqn)
to work so well (e.g. replay buffer, target networks).

### Twin Delayed Deep Deterministic Policy Gradient

[Twin Delayed DDPG][2] (TD3) is an algorithm that improves on DDPG by making a few
slight modifications. Namely, TD3 learns two Q-functions instead of one and uses the
smaller of the two q-values in Bellman terms, it updates the policy less frequently than
the critic, and it adds noise to actions from the target network to smooth out
Q-estimates.