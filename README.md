# Reinforcement_learning_agent_moving_cargos

## Problem Statement:
You have to move several cargos across the grid world into a common desirable rectangle area. There are several cargos, you can move each of them either horizontally or vertically by one cell up or down. The size of the overall world may vary, as well as the placement of the cargo and desirable area. The game ends when all the cargos are in the desirable area and do not overlap.

### Example of the input:
![image](https://user-images.githubusercontent.com/71794972/208931579-566c72e1-2516-4326-875d-241f1a17cd27.png)

# My model and environment representation:
According to the problem statement, the size of the overall world may vary, and the placement of the cargos and desirable area. So, the solution to this problem should not depend on the size of the grid or the objects. To achieve that, I will represent the environment or the state as a tensor with 3 channels, and the model will consist of 4 convolutional layers each of them followed by batch normalization and max pooling layers.
![image](https://user-images.githubusercontent.com/71794972/208931792-f2962a60-73f4-4e4d-a60c-501be109d004.png)

### What does the 3 channels contain?
![image](https://user-images.githubusercontent.com/71794972/208932253-78a5aa12-f26a-4b39-ae9b-b180158e0b60.png)

-	First channel contains ones at the cells of the desirable area
-	Second channel contains ones at the cells of the target cargo
-	Third channel contains ones at the cells of all other cargos


### Why did I use conv layers?

To make the model handle different grid sizes, the conv layers only care about the channel size (which will be fixed according to the environment representation) and it doesn't care about the height and width of the grid. 

### Why did I use batch norm and max pool layers?

I used batch norm to make the training process faster and max pool to increase the receptive field of the model. By increasing the receptive field, the model will be able to see the object and the desirable area at the same time which will help it to make better decisions.

### What is the output of the model?

The number of channels for the final layer of the model is equal to the number of actions the model can do which is 5 because the model can leave the object in the same place, go up, down, right, or left. This 5-channel output will later be fed to a global average polling layer in the forward function to reduce it to only 5 numbers each number representing the action value. 

## Training dataset:
In addition to the random maps generated, I also generated 10 maps by hand. They have varying levels of difficulties in order for the model to generalize. I tried to put the objects and the desirable area in strategic locations in order to achieve the maximum benefits during training the model. 

## Reward function:
-	If action is "rest" and the cargo isn't fully inside the desirable area then return a negative reward, if it is in the desirable area then return 0
-	If the action is not reset, the negative Euclidian distance is calculated between the cargo and the center of the desirable area before and after moving the cargo. If this action moves the cargo farther away from the desirable area, a negative reward is returned, else a positive reward is returned.
-	If the action leads to going outside of the grid a negative reward is returned.
-	For every cargo part inside the desirable area, +1 is added to the reward
-	For every part of cargo overlapping inside the desirable area, -1 is added to the reward
-	If the action that the agent did gave it a reward less than the reward it already has, then a negative reward is returned. This simulate the case when the cargo is inside the desirable area and it is just wondering around without reaching the terminal state.
-	If reached the terminal state where all the cargo parts are inside the desirable area, a positive reward is returned equal to the number of cargo parts

## The way of training the model:
I used deep Q-learning algorithm to train the model. The aim is to train a policy that tries to maximize the discounted cumulative reward R_(t_0 )=∑_(t=t_0)^∞▒〖γ^(t-t_0 ) r_t 〗, where R_(t_0 ) is also known as the return. The discount, γ, should be a constant between 0 and 1 that ensures the sum converges. The main idea behind Q-learning is that if we had a function Q^*:State ×Action→R, that could tell us what our return would be, if we were to take an action in a given state, then we could easily construct a policy that maximizes our rewards:
π^* (s)=argmax_a Q^* (s,a)
However, we don't know everything about the world, so we don't have access to Q^*. But, since neural networks are universal function approximators, we can simply create one and train it to resemble Q^*.
For out training update rule: 
Q^π (s,a)=r+γQ^π (s^',π(s^' ))
The difference between the two sides of the equation is known as the temporal difference error, δ:
δ=Q(s,a)-(r+γ  max┬a⁡〖Q(s^',a))〗
To minimize this error, we will use the Huber loss. The Huber loss acts like the mean squared error when the error is small, but like the mean absolute error when the error is large. This make it more robust to outliers when the estimates of Q are noisy. We calculate this over a batch of transitions, B, sampled from the replay memory:
L=1/(|B|)  ∑_((s,a,s^',r)  ϵ B)▒〖L(δ)〗
where L(δ)={█(1/2 δ^2       for |δ|≤1,@|δ|-1/2     otherwise      )┤

**Actions are selected using epsilon greedy policy**

## The main training loop description:

In the beginning, we choose one of the grids from the training data. There is also a memory replay for each map, this is helpful when I try to optimize the model because I want the model to learn from different grids in the same time and not to run the model for like 100 episodes on the same map and then train it on another map because by doing that, the model will suffer from catastrophic gradient forgetting, so by training the model on different map each pass, it will reduce this problem dramatically and make the model generalize better. Then transform the initial state to our state representatives. Then, an action is sampled for a given cargo and executed, and observe the next state and reward, and optimize the model once. When the episode ends (reached the terminal, or exceeded the maximum number of moves), the loop is restarted with a new grid. The number of episodes here is 100, however, if you increase it, the model will generalize better. Training RL agents can be a noisy process, so restarting training can produce better results if convergence is not observed.

## Conclusion:
In conclusion, the deep q-learning technique was applied and it was able to converge on the examples that it was trained on. To improve this model in the future, you should add more layers to the model and increase the receptive field. Increase the size of the grids that the world can generate. Train it for longer episodes.







