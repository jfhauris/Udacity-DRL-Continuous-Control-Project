# Udacity-DRL-Continuous-Control-Project
Implementation of PPO to perform continuous control of robot arm

# Project 2: Continuous Control of Robot Arm 

## Introduction

For this project I used a the Proximal Poliocy Optimization (PPO) algorithm with Actor-Critic networks to train an agent to remain in close contact with a designated coordinate.  The envirnment used the Unity Reacher environment.  In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## The Proximal Policy Optimization (PPO) Algorithm

The code consists of:
- ppo_agent.py: a PPOAgent which creates the step and learning abilities
- network_model.py: a PPONet which instaniates an actor and critic network
- MainCode.ipynb: which performs initialization and training

Running the MainCode.ipynb will run the training and inference engines of the code.  There were 2 options.  This code implements option 2 which has 20 simulatneous agents running.  An initial effort to run option one (a single agent version) but training was extremely long and not promising.  Because of the 20 agents providing additional information to the learner, training was achieved much faster.  

The problem is solved for the second version of the environment when the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically, after each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
This yields an average score for each episode (where the average is over all 20 agents).



The learning algorithm used is Proximal Policy Optimization (PPO) modified for continous action space. The input to the neural network model is observation vector (33 real numbers). The model consists of 2 seperate neural networks - actor and critic which are interfaced with one class `PPOPolicyNetwork`. 

The actor network takes observation as an input and outputs actions. Due to the `tanh` activation function on the output layer there is no need to scale or clip actions to fit -1, 1 range. The critic network is used to compute advantage returns which requires state value estimation. It outputs 1 real number - state value estimate for the given state.

Action probabilites are taken from the normal distribution.

## Parameters and hyperparameters

### Neural networks

The actor network directly outputs action which agent will take and use without any additional clipping, normalizing or preprocession. That's why it outputs 4 values - size of the action space. The critic network is not directly needed for the PPO algorithm (original paper describes policy network and surrogate function which counts ration of new action probabilites to old ones - actor would suffice) but it's very helpful to compute advantages which requires value for state.

The hidden size parameters was choosen after careful tuning. I started from 64 nodes and after every increase agent took less episodes to converge (while also needed more computing power). 

#### Actor network

- 3 fully connected layers
- 33 input nodes [observation vector size], 4 output nodes [action vector size], 512 hidden nodes in each layer
- ReLU activations, tanh on last layer

#### Critic network

- 3 fully connected layers
- 33 input nodes [observation vector size], 1 output nodes, 512 hidden nodes in each layer
- ReLU activations, no activation on last layer

### Main hyperparameters

- Discount rate - `0.99`
- Tau - `0.95`
- Rollout length - `2048`
- Optimization epochs - `10`
- Gradient clip - `0.2`
- Learning rate - `3e-4`

## Results

![](images/last_reward.png)
![](images/average_reward.png)

The second chart shows an average reward over 100 consecutive episodes. It's steadly increasing and reached required `30+` around 240 episode. Last values printed in the console are:

```
Episode: 250 Total score this episode: 34.51249922858551 Last 100 average: 30.996649307170888
```

and this model is saved as `models/ppo-max-hiddensize-512.pth`

## Next steps

- **Hyperparameter tuning** - I focused on tuning hidden size and gradient clip which gave major improvements. Other parameters would probably impact learning and it's worth to check how.
- **DDPG** -  I gave up on DDPG as it was learning *very* slowly. But it would be good to se how it *actualy* compares with PPO, not just how it feels.
- **Try PPO on other environment** - to see if PPO will be still good.
- **Write generic implementation** - to reuse this repository on other problems and with other libraries (like `Gym`).









### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
Download the Unity Environment
For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

Version 1: One (1) Agent
Linux: click here
Mac OSX: click here
Windows (32-bit): click here
Windows (64-bit): click here
Version 2: Twenty (20) Agents
Linux: click here
Mac OSX: click here
Windows (32-bit): click here
Windows (64-bit): click here
Then, place the file in the p2_continuous-control/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

(For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this link (version 1) or this link (version 2) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)

### Instructions

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 
