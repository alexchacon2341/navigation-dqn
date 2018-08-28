# navigation-dqn
A reinforcement learning project involving an environment in which an agent is trained with a DQN to collect bananas.

OVERVIEW

This repository is an example of deep reinforcement learning using a recurrent neural network (RNN). The coding framework was provided courtesy of Udacity's Deep Reinforcement Learning Nanodegree Program, with an environment created using Unity's ML-Agents plugin.

In the example, an agent is trained using a Deep Q-Network (DQN) over a series of episodes. It begins with no knowledge of its environment, and first learns only by taking actions at random. The agent's experiences are catalogued in a Q-table that tallies expected rewards, and it gradually begins making decisions in order to maximize its reward instead of choosing random actions.The environment is considered solved once the agent achieves an average score of +13 over 100 consecutive episodes. The trained agent whose experiences are included in "nav_weights.pth" achieved an average score of 13.00 in 572 episodes.

ENVIRONMENT

In the environment, an agent is tasked with collecting bananas in a square world. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The goal is therefore to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

0 - move forward.
1 - move backward.
2 - turn left.
3 - turn right.

The environment is very similar to ML-Agents' standard Banana Collector environment. An example of this environment can be seen here: https://www.youtube.com/watch?v=GFSBQ08WmTQ

INSTRUCTIONS

To begin, you will need to set up your Python environment by following the instructions below.

1. Create (and activate) a new environment with Python 3.6.

Linux or Mac:
conda create --name drlnd python=3.6
source activate drlnd

Windows:
conda create --name drlnd python=3.6 
activate drlnd

2. Follow the instructions in this repository to perform a minimal install of OpenAI gym.

Next, install the classic control environment group by following the instructions here.
Then, install the box2d environment group by following the instructions here.

3. Clone the repository (if you haven't already!), and navigate to the python/ folder. Then, install several dependencies.

git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
Create an IPython kernel for the drlnd environment.
python -m ipykernel install --user --name drlnd --display-name "drlnd"
Before running code in a notebook, change the kernel to match the drlnd environment by using the drop-down Kernel menu.
Kernel
