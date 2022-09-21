# Maze solving with different algorithms and their comparisons.
1) With Value Iteraiton and Policy Iteration
2) Q-learning 
3) Double-Q-learning
4) SARSA (State-Action-Reward-State-Action)

## 1) VI and PI

![image](https://github.com/PLEX-GR00T/Maze_solving_MDP/blob/main/Output%20Must%20Watch/iterations.gif)

## 2) Q-learning

<p float="left">
  <img src="https://github.com/PLEX-GR00T/Maze_solving_MDP/blob/main/Output%20Must%20Watch/Q-linear.png" width="400" height="300"/>
  <img src="https://github.com/PLEX-GR00T/Maze_solving_MDP/blob/main/Output%20Must%20Watch/q-exponential.png" width="400" height="300" /> 
</p>

## 3) Double-Q-learning

<p float="left">
  <img src="https://github.com/PLEX-GR00T/Maze_solving_MDP/blob/main/Output%20Must%20Watch/Double%20q-linear.png" width="400" height="300" />
  <img src="https://github.com/PLEX-GR00T/Maze_solving_MDP/blob/main/Output%20Must%20Watch/Double%20q-exponential.png" width="400" height="300" /> 
</p>

## 4) SARSA

<p float="left">
  <img src="https://github.com/PLEX-GR00T/Maze_solving_MDP/blob/main/Output%20Must%20Watch/SARSA%20Linear.png" width="400" height="300" />
  <img src="https://github.com/PLEX-GR00T/Maze_solving_MDP/blob/main/Output%20Must%20Watch/SARSA%20Exponential.png" width="400" height="300" /> 
</p>

# SARSA using Q-learning

You can find the code for results below [here](https://github.com/PLEX-GR00T/Maze_solving_MDP/blob/main/Q-learning_and_SARSA_on_maze.ipynb). In which, we will collect the rewards for 5 runs and plot them together to see any patterns.

<p float="left">
  <img src="https://github.com/PLEX-GR00T/Maze_solving_MDP/blob/main/Output%20Must%20Watch/input%20maze.png" width="280" height="280" />
  <img src="https://github.com/PLEX-GR00T/Maze_solving_MDP/blob/main/Output%20Must%20Watch/output%20maze.png" width="280" height="280"/> 
  <img src="https://github.com/PLEX-GR00T/Maze_solving_MDP/blob/main/Output%20Must%20Watch/Q-learning%20and%20SARASA.png" width="370" height="280" />
</p>

We see the common pattern that the rewards are initially bad, but as the number of episodes increases, the agent gets better and the reward reach an asymptote.
