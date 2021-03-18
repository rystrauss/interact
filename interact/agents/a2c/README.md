# Advantage Actor-Critic

Advantage Actor-Critic (A2C) is a relatively simply actor-critic method which uses the
value function as the baseline in the policy update. A2C is based on
[Asynchronous Advantage Actor-Critic (A3C)](https://arxiv.org/abs/1602.01783), which was
very influential when first proposed. However, as explained in
this [blog post](https://openai.com/blog/baselines-acktr-a2c/), OpenAI found that a
synchronous version of the algorithm, namely A2C, performed just as well while being
simpler.
