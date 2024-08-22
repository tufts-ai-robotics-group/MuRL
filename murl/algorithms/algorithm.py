#!/usr/bin/env python3

"""
Plugin system for algorithms
"""

import abc
from gymnasium import Env
import numpy as np
import logging
from gymnasium import Env


class Algorithm(metaclass=abc.ABCMeta):
    """
    Abstract base class, contains spec for RL algoriths.
    """

    @abc.abstractmethod
    def update(self) -> None:
        """
        Update model
        """

        ...

    @abc.abstractmethod
    def run_episode(self, max_timesteps: int, initial_state, validation: bool) -> None:
        """
        Run single episode
        """

        ...

    @abc.abstractmethod
    def save_model(self) -> None:
        """
        Placeholder method to save model
        """

        ...

    def load_model(self) -> None:
        """
        Placeholder method to load a previously created model
        """

        ...


class PolicyGradientAlgorithm(Algorithm):

    def __init__(
        self,
        random_number_generator: np.random.Generator,
        logger: logging.Logger,
        env: Env,
        **kwargs
    ) -> None: ...

    @abc.abstractmethod
    def select_action(self) -> None:
        """
        Placeholder for action selection of policy gradient
        algorithm
        """

        pass


class ActionValueAlgorithm(Algorithm):
    """
    abstract base class, contains spec for rl algoriths.
    """

    class Policy(metaclass=abc.ABCMeta):
        """
        Abstract base class, contains spec for RL algoriths.
        """

        def __init__(
            self,
            random_number_generator: np.random.Generator,
            logger: logging.Logger,
            env: Env,
            **kwargs
        ) -> None:
            import rl.algorithms.action_value as policies

            self._logger = logger
            self._env = env
            self._config = kwargs
            self._random_number_generator = random_number_generator

        @abc.abstractmethod
        def select_action(self) -> list:
            """
            Placeholder method responsible for selecting an action
            """

            ...

    def __init__(
        self,
        random_number_generator: np.random.Generator,
        logger: logging.Logger,
        env: Env,
        **kwargs
    ) -> None:
        import rl.algorithms.action_value as policies

        self._logger = logger
        self._env = env
        self._config = kwargs
        self._random_number_generator = random_number_generator

        self._training_policy = policies.action_value_policies[
            kwargs["training_policy"]["name"]
        ](
            random_number_generator,
            logger,
            env,
            **kwargs["training_policy"],
        )

        self._validation_policy = policies.action_value_policies[
            kwargs["validation_policy"]["name"]
        ](
            random_number_generator,
            logger,
            env,
            **kwargs["validation_policy"],
        )
