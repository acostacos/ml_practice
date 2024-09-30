from __future__ import annotations

from collections import defaultdict

import gymnasium as gym
import numpy as np
from tqdm import tqdm

class QLearningAgent():
    def __init__(
        self,
        env: gym.Env,
        n_episodes: int,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Q-Learning Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon. Q-Learning = Active RL.
        Uses epsilon-greedy scheme for exploration.

        Args:
            n_epsiodes: Number of episodes to train on
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.training_error = []

        self.n_episodes = n_episodes
        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    def train(self, env: gym.Env):

        for _ in tqdm(range(self.n_episodes)):
            # Set initial state
            obs, _ = env.reset()
            done = False

            # Play one episode
            while not done:
                # Take an action
                action = self.get_action(env, obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)

                # Update the agent
                self.update(obs, action, reward, terminated, next_obs)

                # Update if the environment is done and the current obs
                done = terminated or truncated
                done = next_obs
            
            # Decay epsilon (less exploration)
            self.decay_epsilon()

    def get_action(self, env: gym.Env, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        Epsilon-greedy scheme for exploration.
        """
        # With probability epsilon, return a random action to explore the environment (explore)
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # With probability (1 - epsilon), act greedily (exploit)
        return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool]
    ):
        """Updates the Q-value of an action"""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)
    
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

