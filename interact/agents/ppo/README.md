[PPO]: https://arxiv.org/abs/1707.06347
[GAE]: https://arxiv.org/abs/1506.02438

# Proximal Policy Optimization

[Proximal Policy Optimization][PPO] (PPO) is an actor-critic method. A key challenge when using deep neural networks as
function approximators in RL is instability: even modest parameter updates can often cause very dramatic changes
to the policy and cause learning to collapse. PPO addresses this problem by using a novel clipping term in the
objective function which ensures that network updates seldom move the new policy far from the previous policy.

PPO often employs one other corrective in the form of [generalized advantage estimation][GAE]: rather than using an
estimate of the Q-function in the actor-critic objective, an estimate of the advantage function is computed instead.
Intuitively, this is a lower-variance estimate of the Q-function that seeks to uncover actions that
have unusually high (or low) payoffs in comparison to a random action in a given state.