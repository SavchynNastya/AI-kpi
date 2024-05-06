from typing import Dict, List
import numpy as np
import pandas as pd


NUM_EPISODES = 1000
DISCOUNT_FACTOR = 0.9
LEARNING_RATE = 0.1


class QLearning:
    def __init__(self, num_states: int, adjacency: Dict[int, List[int]], learning_rate: float = LEARNING_RATE, 
                 discount_factor: float = DISCOUNT_FACTOR, num_episodes: int = NUM_EPISODES):
        self.num_states = num_states
        self.adjacency = adjacency
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.num_episodes = num_episodes
        self.Q = np.zeros((num_states, num_states))

    def train(self) -> np.ndarray:
        for _ in range(self.num_episodes):
            state = 0
            while state != 6:
                action = np.random.choice(self.adjacency[state])
                next_state_max_q = np.max(self.Q[action])
                reward = 1 if action == 6 else 0
                self.Q[state, action] = (1 - self.learning_rate) * self.Q[state, action] + \
                                         self.learning_rate * (reward + self.discount_factor * next_state_max_q)
                state = action
        return self.Q

    def find_optimal_path(self) -> List[int]:
        optimal_path = [0]
        state = 0
        while state != 6:
            action = np.argmax(self.Q[state])
            optimal_path.append(action)
            state = action
        return optimal_path


adjacency = {
    0: [7, 1, 3],
    1: [0, 2],
    2: [1, 3, 4, 5],
    3: [0, 2, 4, 7],
    4: [2, 3, 5, 7],
    5: [2, 4, 6],
    6: [5],
    7: [0, 3, 4]
}

q_learning = QLearning(num_states=8, adjacency=adjacency)

Q_matrix = q_learning.train()

print("Q matrix:")
df = pd.DataFrame(Q_matrix, columns=[i for i in range(8)], index=[i for i in range(8)])
print(df)

optimal_path = q_learning.find_optimal_path()
print("Optimal path:", optimal_path)
