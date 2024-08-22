#!/usr/bin/env python3n

"""
A simple tabular Q Learning algorithm
"""

import numpy as np
import pickle
from datetime import datetime
from gymnasium import Env

import logging
from rl.algorithms.algorithm import ActionValueAlgorithm


class QLearning(ActionValueAlgorithm):
    """
    Includes the methods needed for the q learning
    """

    def __init__(
        self,
        random_number_generator: np.random.Generator,
        logger: logging.Logger,
        env: Env,
        **kwargs,
    ) -> None:
        super().__init__(random_number_generator, logger, env, **kwargs)

        self._gamma = int(self._config["gamma"])
        self._alpha = int(self._config["alpha"])

        self._q = {}

    def update(self, timestep_tracker: dict) -> None:
        """
        update function calculates new q values

        :param timestep_tracker: a dict containing lists of rewards for each state action combination
        """
        # seperate dict
        current_state_action_tuple = timestep_tracker["state_action_current"]
        next_state = timestep_tracker["state_next"]
        reward = timestep_tracker["reward"]

        q_current = (
            self._q[current_state_action_tuple]
            if current_state_action_tuple in self._q.keys()
            else 0.0
        )

        # Get q values for next state
        q_values = [
            self._q[state_action_pair]
            for state_action_pair in self._q
            if state_action_pair[0] == next_state
        ]

        # Check if q_values exists for next state, if so choose max q, if not q_next is zero
        q_next = np.max(q_values) if len(q_values) > 0 else 0.0

        # Update q
        self._q[current_state_action_tuple] = q_current + (
            self._alpha * (reward + (self._gamma * q_next) - q_current)
        )

    def run_episode(
        self, validation: bool, initial_state: tuple, total_timestep_counter: int
    ):
        """Runs a single episode, can be used for training or validation

        :param validation: if episode is validation.
        """

        episode_rewards = []
        current_observation = tuple(initial_state)

        timestep_counter = total_timestep_counter

        complete = False

        while not complete:
            timestep_counter += 1

            action = (
                self._validation_policy.select_action(self._q, current_observation)
                if validation
                else self._training_policy.select_action(self._q, current_observation)
            )

            next_observation, reward, complete, _, _ = self._env.step(action)
            next_observation = tuple(
                next_observation
            )  # Convert to tuple so that is hashable

            episode_rewards.append(reward)

            if not validation:
                self.update(
                    {
                        "state_action_current": (current_observation, action),
                        "state_next": next_observation,
                        "reward": reward,
                    }
                )

            current_observation = next_observation

        return np.sum(episode_rewards), len(episode_rewards), timestep_counter

    def save_model(self, location: str = "") -> None:
        """
        Saves model to a specified location, stores as pkl

        :param location: location to store pkl
        """

        location = (
            location
            if location != ""
            else f"{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}_q_learning.pkl"
        )

        with open(location, "wb") as file:
            pickle.dump(self._q, file)

    def load_model(self, location: str) -> None:
        """
        Loads Q table with previously saved model

        :param load_model: where to load saved model from
        """

        with open(location, "rb") as file:
            self._q = pickle.load(file)
