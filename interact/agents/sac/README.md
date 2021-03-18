[1]: https://arxiv.org/abs/1801.01290

[2]: https://arxiv.org/abs/1812.05905

# Soft Actor-Critic

[Soft Actor-Critic][1] is a stochastic, off-policy, actor-critic method that is based on
maximum entropy RL. The policy is trained to maximize a balance between the standard
expected return and the policy's entropy, a measure of its randomness. This helps
encourage exploration, prevent local optima early in training, and leads to good sample
efficiently and robustness.

This particular implementation uses the more modern version of the algorithm,
described [here][2].