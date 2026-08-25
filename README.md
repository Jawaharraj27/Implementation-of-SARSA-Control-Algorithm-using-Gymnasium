# Implementation-of-SARSA-Control-Algorithm-using-Gymnasium

## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

---

## Problem Statement

Implement the SARSA (State-Action-Reward-State-Action) control algorithm in the Gymnasium FrozenLake-v1 environment. The agent must learn the optimal action-value function using an epsilon-greedy policy and gradually improve its performance by balancing exploration and exploitation.

## Software Requirements


## Environment Description

The FrozenLake-v1 environment represents a $4 \times 4$ grid containing:

S – Starting state
F – Frozen/safe surface
H – Hole
G – Goal

The agent starts at state S and must reach G without falling into a hole.

There are 16 states and 4 possible actions.

Action	Meaning
0	Left
1	Down
2	Right
3	Up

For this implementation, is_slippery=False is used so that the environment is deterministic.

The reward structure is:

+1 for reaching the goal
0 for other transitions, including falling into a hole

## Theory

SARSA stands for:

$$
S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
$$

It updates the Q-value using the action actually selected in the next state.

The SARSA update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $A_{t+1}$ | Next action selected using the current policy |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |

---

## Epsilon-Greedy Policy

SARSA uses an epsilon-greedy policy for action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---


## Algorithm
Initialize the FrozenLake environment.
Initialize the Q-table with zeros.
Set the learning parameters $\alpha$, $\gamma$, and $\epsilon$.
Reset the environment and obtain the initial state.
Select the initial action using the epsilon-greedy policy.
Perform the selected action.
Observe the reward and next state.
Select the next action using the epsilon-greedy policy.
Update the Q-value using the SARSA update equation.
Set the next state and next action as the current state and action.
Repeat until the episode terminates.
Decrease epsilon gradually to reduce exploration.
Repeat the process for the specified number of episodes.
Extract the learned policy using the maximum Q-value for each state.
Calculate the state-value function.
Display the final Q-table, value function, learned policy, and average reward.
Plot the learning curve.

## Python Program

```python

# -------------------------------------------------
# SARSA Training
# -------------------------------------------------
# Write your code here
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=False
)

n_states = env.observation_space.n
n_actions = env.action_space.n


# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 15000
max_steps_per_episode = 100

alpha = 0.1
gamma = 0.95

epsilon = 0.8
epsilon_min = 0.01
epsilon_decay = 0.9997


# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

# Initial Q-values are NOT zero
Q = np.full(
    (n_states, n_actions),
    0.1
)


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):

    if np.random.random() < epsilon:
        return env.action_space.sample()

    return np.argmax(Q[state])


# -------------------------------------------------
# SARSA Training
# -------------------------------------------------

episode_rewards = []

for episode in range(num_episodes):

    state, info = env.reset()

    # Select initial action
    action = epsilon_greedy_action(
        state,
        epsilon
    )

    total_reward = 0

    for step in range(max_steps_per_episode):

        next_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        # Terminal state
        if terminated or truncated:

            Q[state, action] = Q[state, action] + alpha * (
                reward - Q[state, action]
            )

            break

        # Select next action
        next_action = epsilon_greedy_action(
            next_state,
            epsilon
        )

        # SARSA update
        Q[state, action] = Q[state, action] + alpha * (
            reward
            + gamma * Q[next_state, next_action]
            - Q[state, action]
        )

        state = next_state
        action = next_action

    episode_rewards.append(total_reward)

    # Reduce epsilon
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


# -------------------------------------------------
# State-Value Function
# -------------------------------------------------

state_values = np.max(Q, axis=1)


# -------------------------------------------------
# Learned Policy
# -------------------------------------------------

learned_policy = np.argmax(Q, axis=1)


# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(values):

    print("\nEstimated State-Value Function:")

    print(
        np.round(
            values.reshape(4, 4),
            4
        )
    )


def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [
            action_symbols[action]
            for action in policy
        ]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)


# -------------------------------------------------
# Output
# -------------------------------------------------

print("Name : JAWAHAR RAJ N")
print("Register Number : 212223240057")

print("\nFinal Q-table:")
print(np.round(Q, 4))

print_value_function(state_values)

print_policy(learned_policy)

average_reward = np.mean(
    episode_rewards[-1000:]
)

print(
    "\nAverage reward over last 1000 episodes:",
    average_reward
)


# -------------------------------------------------
# Learning Curve
# -------------------------------------------------

window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))

plt.plot(moving_average)

plt.xlabel("Episode")
plt.ylabel("Average Reward")

plt.title(
    "SARSA Learning Curve - FrozenLake"
)

plt.grid(True)

plt.show()


env.close()



```
---

## Output


Final Q-table:

<img width="541" height="455" alt="image" src="https://github.com/user-attachments/assets/6c31f70a-ea08-4ce3-bc7b-e3cfb76ccce2" />




Estimated State-Value Function:

<img width="403" height="123" alt="image" src="https://github.com/user-attachments/assets/0b63c02d-7f0d-495d-a018-87aaec297849" />




Learned Policy:

<img width="288" height="121" alt="image" src="https://github.com/user-attachments/assets/a986f072-8ba8-4664-bd94-13c222222998" />




Average reward over last 1000 episodes: 

<img width="515" height="47" alt="image" src="https://github.com/user-attachments/assets/51cab6a8-d6b4-47c3-9f67-478af4014810" />

<img width="997" height="627" alt="image" src="https://github.com/user-attachments/assets/f2efb6b6-6ee3-4a14-8878-b6b2b588cf85" />
```


---

## Result

The SARSA control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The agent learned an action-value function through repeated interaction with the environment and developed a policy for selecting actions that lead toward the goal while avoiding holes.





## Inference

The experiment demonstrates that SARSA can learn an effective policy through on-policy temporal-difference learning. Initially, the agent explores the environment using a high epsilon value. As training progresses, epsilon decreases, allowing the agent to exploit the learned Q-values more frequently. The increasing average reward indicates that the agent gradually improves its ability to reach the goal.`


