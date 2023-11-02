import numpy as np


class SARSAAgent:
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.Q = {}  # Initialize Q-values as an empty dictionary

        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.1

    def get_Q(self, state, action):
        # Get Q-value for state-action pair, with default value of 0
        return self.Q.get((tuple(state), action), 0)

    def set_Q(self, state, action, value):
        # Set Q-value for state-action pair
        self.Q[(tuple(state), action)] = value

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            # Explore: select a random action
            return np.random.choice(self.action_space.n)
        else:
            # Exploit: select the action with max Q-value
            q_values = [self.get_Q(state, action) for action in range(self.action_space.n)]
            return np.argmax(q_values)

    def update_Q(self, state, action, reward, next_state, next_action):
        td_target = reward + self.gamma * self.get_Q(next_state, next_action)
        td_error = td_target - self.get_Q(state, action)
        self.set_Q(state, action, self.get_Q(state, action) + self.alpha * td_error)
