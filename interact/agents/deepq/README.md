[1]: https://www.nature.com/articles/nature14236
[2]: https://arxiv.org/abs/1710.02298
[3]: https://arxiv.org/abs/1509.06461
[4]: https://arxiv.org/abs/1511.06581
[5]: https://arxiv.org/abs/1511.05952

# Deep Q-Learning

The deep Q-network (DQN) agent, originally proposed by [Mnih et. al.][1], was introduced as the first agent capable of
achieving a range of human-level competencies on numerous challenging tasks.

DQN approximates the action-value function (or Q-function) with a deep convolutional neural network, which uses
hierarchical layers of convolutional filters to build progressively abstract data representations from raw pixel input.

This implementation also contains some improvements on the original DQN algorithm.

## DQN Variants

This implementation also includes the following improvements on the original DQN algorithm (a review of all DQN
modifications can be found [here][2]).

### Double DQN

Vanilla Q-learning has been shown to overestimate Q-values due to the max operator in the update rule, and DQN
suffers from substantial overestimations  in some settings. [Double DQN][3] (DDQN) is an attempt to remedy that that
leads to better performance on several Atari games.
 
### Prioritized Replay Experience

[Prioritized Replay Experience][5] replaces the standard experience replay
(which is sampled uniformly at random) with one that replays important
transitions more frequently. This allows the agent to learn more
efficiently and leads to much better performance than vanilla DQN.