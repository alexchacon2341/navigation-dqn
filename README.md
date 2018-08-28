# navigation-dqn
A Reinforcement Learning project involving an environment in which an agent was trained to find yellow bananas while avoiding blue bananas with a DQN.

OVERVIEW

This repository is an example of deep machine learning. The coding framework was provided courtesy of Udacity's Deep Reinforcement Learning Nanodegree Program, with an environment provided by Unity's ML-Agents plugin.

In the environment, an agent is tasked with collecting bananas in a square world. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The goal, therefore, is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

0 - move forward.
1 - move backward.
2 - turn left.
3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

The environment was created for Unity's ML-Agents plugin, and is very similar to its standard Banana Collector environment. An example of this environment can be seen here: https://www.youtube.com/watch?v=GFSBQ08WmTQ

The agent is trained using a reinforcement learning structure known as a Deep Q-Network (DQN) over a series of episodes. The agent begins with no knowledge of its environment, and at first learns only by taking actions at random. Its experiences are catalogued in a Q-table that tallies expected rewards, and the agent gradually begins making decisions to maximize its reward instead of choosing actions at random. The environment is considered solved once the agent achieves an average score of +13 over 100 consecutive episodes. When training the agent with the included 
