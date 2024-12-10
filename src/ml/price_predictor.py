"""
Price Prediction Environment and Hybrid Price Predictor

This module implements a custom OpenAI Gym environment for price prediction tasks and a hybrid 
price prediction model combining clustering, dimensionality reduction, and reinforcement learning.

Classes:
    PricePredictionEnvironment: A custom Gym environment simulating price prediction scenarios.
    HybridPricePredictor: A hybrid model that integrates clustering, PCA, and Deep Q-Learning.

Dependencies:
    - numpy
    - pandas
    - scikit-learn
    - tensorflow
    - gym
    - collections (deque)
    - random

Version: 1.0.0
Author: [Your Name]
Last Modified: [Date]

Usage Warnings:
    This module assumes the input data is preprocessed and normalized appropriately.
    The environment and model are designed for educational purposes and require further 
    customization for production use.

Performance Characteristics:
    The reinforcement learning model's performance depends on the quality of the input data,
    hyperparameter tuning, and the training process.

Memory Considerations:
    The HybridPricePredictor uses a memory buffer (deque) to store past experiences. Ensure 
    sufficient memory is available for larger buffer sizes.

Relevant Documentation Links:
    - OpenAI Gym: https://www.gymlibrary.dev/
    - Scikit-learn: https://scikit-learn.org/
    - TensorFlow: https://www.tensorflow.org/
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
import gym
from collections import deque
import random

# Suppress numpy deprecation warnings
np.warnings = np.testing.suppress_warnings()

class PricePredictionEnvironment(gym.Env):
    """
    Custom Gym Environment for Price Prediction.

    This environment simulates a price prediction scenario where an agent can take actions 
    to predict whether the price will increase, decrease, or stay the same.

    Attributes:
        data (np.ndarray): The input dataset representing price data.
        current_step (int): The current step in the dataset.
        action_space (gym.spaces.Discrete): Action space with three possible actions 
            (Increase, Decrease, Hold).
        observation_space (gym.spaces.Box): Observation space representing the state at 
            each step.

    Parameters:
        data (np.ndarray): A 2D numpy array where each row represents a state, and the 
            last column represents the price.

    Notes:
        - The dataset must be preprocessed and normalized before being passed to this class.
        - The environment assumes that `data` contains at least two rows.
    
    Example Usage:
        data = np.random.rand(100, 5)  # Example dataset with 100 rows and 5 features
        env = PricePredictionEnvironment(data)
        state = env.reset()
        next_state, reward, done, info = env.step(action=0)
    """

    def __init__(self, data):
        """
        Initialize the PricePredictionEnvironment.

        Args:
            data (np.ndarray): A 2D numpy array where each row represents a state, and the 
                last column contains the price values.
        """
        super().__init__()
        self.data = data
        self.current_step = 0

        # Define action space: 0 -> Increase, 1 -> Decrease, 2 -> Hold
        self.action_space = gym.spaces.Discrete(3)

        # Define observation space: Continuous values for each feature in the dataset
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(data.shape[1],), dtype=np.float32
        )

    def step(self, action):
        """
        Take an action in the environment and observe the result.

        Args:
            action (int): The action taken by the agent (0: Increase, 1: Decrease, 2: Hold).

        Returns:
            tuple:
                - next_state (np.ndarray): The next state after taking the action.
                - reward (float): The reward obtained from taking the action.
                - done (bool): Whether the episode has ended.
                - info (dict): Additional information (currently empty).
        
        Raises:
            IndexError: If `current_step + 1` exceeds the dataset length.
        """
        reward = self._calculate_reward(action)
        self.current_step += 1

        # Check if the episode is done
        done = self.current_step >= len(self.data) - 1

        # Get the next state
        next_state = self._get_state()

        return next_state, reward, done, {}

    def reset(self):
        """
        Reset the environment to the initial state.

        Returns:
            np.ndarray: The initial state of the environment.
        """
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        """
        Retrieve the current state from the dataset.

        Returns:
            np.ndarray: The current state.
        
        Notes:
            The state includes all features of the current row in `data`.
        """
        return self.data[self.current_step]

    def _calculate_reward(self, action):
        """
        Calculate the reward based on the action taken.

        Args:
            action (int): The action taken by the agent.

        Returns:
            float: The reward for the given action.

        Notes:
            - Reward is +1.0 for correct predictions and -1.0 otherwise.
            - A "Hold" action is rewarded if the price change is negligible (< 0.01).
        
        Raises:
            IndexError: If `current_step + 1` exceeds the dataset length.
        """
        current_price = self.data[self.current_step][-1]
        next_price = self.data[self.current_step + 1][-1]
        price_change = next_price - current_price

        # Reward logic based on action and price change
        if (action == 0 and price_change > 0) or \
           (action == 1 and price_change < 0) or \
           (action == 2 and abs(price_change) < 0.01):
            return 1.0
        return -1.0


class HybridPricePredictor:
    """
    Hybrid Model for Price Prediction combining clustering, PCA, and Deep Q-Learning.

    Attributes:
        scaler (StandardScaler): Scaler for normalizing input data.
        kmeans (KMeans): KMeans clustering model with 5 clusters.
        pca (PCA): PCA model for dimensionality reduction to 2 components.
        memory (deque): Experience replay buffer for storing past experiences.
        batch_size (int): Size of batches used during training.
        gamma (float): Discount factor for future rewards in Q-Learning.
        epsilon (float): Exploration rate for ε-greedy policy.
        epsilon_min (float): Minimum exploration rate.
        epsilon_decay (float): Decay rate for exploration probability.
        model (keras.Model): Neural network model for Q-Learning.

    Parameters:
        learning_rate (float): Learning rate for the neural network. Default is 0.001.
        memory_size (int): Maximum size of the experience replay buffer. Default is 2000.
        batch_size (int): Batch size used during training. Default is 32.

    Notes:
        This class integrates unsupervised learning techniques (KMeans and PCA) with 
        reinforcement learning for price prediction tasks.

    Example Usage:
        predictor = HybridPricePredictor()
    """

    def __init__(self, learning_rate=0.001, memory_size=2000, batch_size=32):
        """
        Initialize the HybridPricePredictor.

        Args:
            learning_rate (float): Learning rate for the neural network. Default is 0.001.
            memory_size (int): Maximum size of the experience replay buffer. Default is 2000.
            batch_size (int): Batch size used during training. Default is 32.
        
        Raises:
            ValueError: If `learning_rate`, `memory_size`, or `batch_size` are invalid.
        
        Notes:
            This constructor initializes unsupervised learning models and a deep Q-network.
        """
        
        # Initialize preprocessing models
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=5)
        self.pca = PCA(n_components=2)

        # Initialize experience replay buffer
        self.memory = deque(maxlen=memory_size)

        # Set hyperparameters
        self.batch_size = batch_size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Deep Q-Network
        self.model = self._build_model(learning_rate)
        
    def _build_model(self, learning_rate):
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(3, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                     loss='mse')
        return model
    
    def preprocess_data(self, data):
        scaled_data = self.scaler.fit_transform(data)
        clusters = self.kmeans.fit_predict(scaled_data)
        reduced_data = self.pca.fit_transform(scaled_data)
        return scaled_data, clusters, reduced_data
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(3)
        q_values = self.model.predict(state.reshape(1, -1))
        return np.argmax(q_values[0])
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        
        targets = rewards + self.gamma * \
                 np.amax(self.model.predict_on_batch(next_states), axis=1) * \
                 (1 - dones)
        targets_full = self.model.predict_on_batch(states)
        
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets
        
        self.model.fit(states, targets_full, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay