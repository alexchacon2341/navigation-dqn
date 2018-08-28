[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135612-cbff24aa-7d12-11e8-9b6c-2b41e64b3bb0.gif "Trained Agent"
[image2]: https://lh3.googleusercontent.com/-a0Pu8tBE66A/W4TaVpHShPI/AAAAAAAAF6I/QNUld4w_-7AvXoN7J0TBkDb764qUwKcYACL0BGAs/w530-d-h42-n-rw/Screen%2BShot%2B2018-08-28%2Bat%2B1.07.44%2BAM.png "Action Value Function"

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
approximate the optimal action-value function

![Action Value Function][image2]

which is the maximum sum of rewards r^t discounted by γ at each timestep
*t*, achievable by a behaviour policy π = P(a|s), after making an
observation (s) and taking an action (a).
