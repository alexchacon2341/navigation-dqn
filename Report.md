[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135612-cbff24aa-7d12-11e8-9b6c-2b41e64b3bb0.gif "Trained Agent"
[image2]: https://lh3.googleusercontent.com/-a0Pu8tBE66A/W4TaVpHShPI/AAAAAAAAF6I/QNUld4w_-7AvXoN7J0TBkDb764qUwKcYACL0BGAs/w530-d-h42-n-rw/Screen%2BShot%2B2018-08-28%2Bat%2B1.07.44%2BAM.png "Action Value Function"
[image3]: https://lh3.googleusercontent.com/-OU0OBi7f0L4/W4TdiuPe4oI/AAAAAAAAF68/V9DJFw4fufERS5UfARVIFcRJNdkDogZigCL0BGAs/w530-d-h79-n-rw/Screen%2BShot%2B2018-08-28%2Bat%2B1.28.17%2BAM.png "Loss Function"
[image4]: https://lh3.googleusercontent.com/-y8LZqmVuCW8/W4ToZiIV8bI/AAAAAAAAF7s/21hHC4Z9KKQZBwalr52NQyn9LLRCoiZPACL0BGAs/w530-d-h260-n-rw/Screen%2BShot%2B2018-08-28%2Bat%2B2.14.30%2BAM.png "Hyperparameters"
[image5]: https://lh3.googleusercontent.com/-OxVI2XxvLB0/W4TpYJ8cyLI/AAAAAAAAF8M/-mo-HiacK9UvR95d_g9rnffMTkjv4ZtxgCL0BGAs/w530-d-h357-n-rw/Screen%2BShot%2B2018-08-28%2Bat%2B1.35.00%2BAM.png "Plot"

# Report

### Methodology

The project uses recent advances in training deep neural networks to
develop a novel artificial agent, termed a deep Q-network (DQN), that can
learn successful policies directly from high-dimensional sensory inputs
using end-to-end reinforcement learning. The architecture used is a deep recurrent
neural network (RNN), which uses ____ to ____.

The agent interacts with its environment through a sequence of observations, 
actions and rewards. The agent's goal is to select actions in a fashion that 
maximizes cumulative future reward. More formally, an RNN is used to
approximate the optimal action-value function:

![Action Value Function][image2]

This function produces the maximum sum of rewards *r<sub>t</sub>* discounted by *γ* at each timestep
*t*, achievable by a behaviour policy *π = P(a|s)*, after making an
observation *(s)* and taking an action *(a)*.

In order to correct for instability and even divergence when a nonlinear function approximator is used to represent the action-value function, a replay buffer is implemented. This buffer has two primary facets:

1. An experience replay mechanism that randomizes over the dta, removing correlations in the observation space and smoothing over changes in the data distribution.

2. An iterative update that adjusts the action-values towards target values that are only periodically updated, reducing correlations with the target.

To perform experience replay, the agent's experiences *e<sub>t</sub> = (s<sub>t</sub>, a<sub>t</sub>, r<sub>t</sub>, s<sub>t + 1</sub>)* are stored at each time-step *t* in a data set *D = {e<sub>1,...,</sub> e<sub>t</sub>}*. Q-learning updates are applied during training on samples of experience *(s,a,r,s')~U(D)*, drawn uniformly at random from the pool of stored samples. The Q-learning update at iteration *i* uses the following loss function:

![Loss Function][image3]

In this function, *γ* is the discount factor determining the agent's horizon, *θ<sub>i</sub>* are the parameters of the Q-network at iteration *i* and *θ<sub>i</sub><sup>-</sup>* are the network parameters used to compute the target at iteration *i*. The target network
parameters *θ<sub>i</sub><sup>-</sup>* are only updated with the Q-network parameters *(θ<sub>i</sub>)* every *C* steps and are held fixed between individual updates.

### Hyperparameters

To best compare across environments, the hyperparemeters used were similar to those used in the [paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) on which the example was based. The algorithm from this research was able to a achieve a level of performance comparable to that of a professional human games tester across a set of 49 Atari games using the same hyperparameters, and these hyperparameters were used in the example to attempt similar results while using an RNN. The exact values and descriptions for each hyperparameter were as follows:

![Hyperparameters][image4]

Using these settings, the environment was solved in 497 episodes with an average consecutive reward of +13.01. The following plot shows the agent's progress throughout the training session:

![Plot][image5]

### Suggestions

While the agent was able to converge on a policy that solved the environment in a relatively short period of time, improvements may still occur with additional changes. To increase the likelihood that the agent continues exploring different actions until the optimal policy has been found, it may be beneficial to implement a curriculum-type structure with gamma, ensuring its decay pauses at certain thresholds until a certain average reward is reached. Increasing the number of layers may also yield better results. Finally, extensions of the DQN algorithm, including Double DQN, Dueling DQN, or Rainbow may converge on a more optimal policy in a shorter period.
