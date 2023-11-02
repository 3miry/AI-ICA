import gym
import numpy as np
from drone_env import DroneEnv
from sarsa_agent import SARSAAgent

env = DroneEnv()
agent = SARSAAgent(env.action_space, env.observation_space)

num_episodes = 1000  # or whatever number of episodes you want to run

for episode in range(num_episodes):
    state = env.reset()
    action = agent.select_action(state)
    done = False
    
    while not done:
        next_state, reward, done, info = env.step(action)
        next_action = agent.select_action(next_state)
        agent.update_Q(state, action, reward, next_state, next_action)
        state, action = next_state, next_action
        env.render()  # render the environment to visualize
        

    
# Save Q-table
np.save('q_table.npy', agent.Q)

# Load Q-table
agent.Q = np.load('q_table.npy', allow_pickle=True)


