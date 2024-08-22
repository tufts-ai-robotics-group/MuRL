#!/usr/bin/env python3

"""
Epsilon Greedy policy

Referances:
    - https://www.geeksforgeeks.org/python-get-key-with-maximum-value-in-dictionary/
"""

import numpy as np
from gymnasium import Env
import logging

from rl.algorithms.algorithm import ActionValueAlgorithm


class EpsilonGreedy(ActionValueAlgorithm.Policy):
    def __init__(
        self,
        random_number_generator: np.random.Generator,
        logger: logging.Logger,
        env: Env,
        **kwargs
    ) -> None:
        super().__init__(random_number_generator, logger, env, **kwargs)
        self._epsilon = self._config["epsilon"]

    def select_action(self, q_table: dict, current_state: tuple) -> int:
        """function to select an action according to the selected policy

        :param q_values: q values
        :param current_state: the current state of the agent
        :return: which action to take
        """
        sample = self._random_number_generator.random()

        # Get dict of action : estimate for each action available in the state
        q_dict = {
            key[1]: value for key, value in q_table.items() if key[0] == current_state
        }

        if sample > self._epsilon and len(q_dict) > 0:
            actions = list(q_dict.keys())
            q_values = list(q_dict.values())

            # Get list of index's with highest q values
            idx = np.argwhere(q_values == np.max(q_values)).flatten()

            # Randomly break ties
            return actions[self._random_number_generator.choice(idx)]

        # Sample from action space
        return self._env.action_space.sample()
