#!/usr/bin/env python3

"""
Greedy policy

Referances:
    - https://www.geeksforgeeks.org/python-get-key-with-maximum-value-in-dictionary/
"""

import numpy as np
from gymnasium import Env
import logging

from rl.algorithms.algorithm import ActionValueAlgorithm


class Greedy(ActionValueAlgorithm.Policy):
    def __init__(
        self,
        random_number_generator: np.random.Generator,
        logger: logging.Logger,
        env: Env,
        **kwargs
    ) -> None:

        super().__init__(random_number_generator, logger, env, **kwargs)

    def select_action(self, q_table: dict, current_state: tuple) -> int:
        """function to select an action according to the selected policy

        :param q_values: q values
        :param current_state: the current state of the agent
        :return: which action to take
        """

        if len(q_table) == 0:
            raise ValueError("Greedy cannot be used if Q values are empty")

        actions = [
            state_action_pair[1]
            for state_action_pair in q_table
            if state_action_pair[0] == current_state
        ]

        q_values = [
            q_table[state_action_pair]
            for state_action_pair in q_table
            if state_action_pair[0] == current_state
        ]

        max_return = np.max(q_values)
        index = np.argwhere(q_values == max_return).flatten()
        action_index = self._random_number_generator.choice(
            index
        )  # If multiple q_values have the same estimated return, randomly break the tie
        return actions[action_index]
