[1]: https://arxiv.org/pdf/2009.04416.pdf

# Phasic Policy Gradient

[Phasic Policy Gradient][1] is an actor-critic reinforcement learning algorithm that separates the training of the
policy and value function into two distinct phases. During the policy phase, the policy is advanced and a disjoint
value function is optimized (i.e. there is no weight sharing between the policy and value networks).  During the
auxiliary phase, an auxiliary value head, which is attached to the policy network, is optimized while a behavior
cloning objective works to keep the policy from changing. This aims to distill useful features from the value function
into the policy network. [Proximal Policy Optimization](../ppo) is used as the driving method during the
policy phase.