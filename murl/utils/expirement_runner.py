#!/usr/bin/env python3

"""
Runner file to parse and execute expirements based on config files
"""

import numpy as np

import os
import logging
import json

from multiprocessing import Pool
import gymnasium as gym

import murl.algorithms as algorithms

from murl import stamped_path

logger = logging.getLogger(__name__)


class ExpirementRunner:
    def __init__(
        self,
        env: gym.Env,
        config_location: str,
        expirement_save_location: str,
        parallel_expirements: int = 0,
        **kwargs,
    ) -> None:
        self._env = env

        self._expirement_save_location = (
            kwargs["expirement_save_location"]
            if "expirement_save_location" in kwargs
            else stamped_path
        )

        # TODO: How to handle callbacks?
        self._callbacks = {}
        self.callbacks["env_instantiation"] = (
            kwargs["callbacks"]["env_instantiation"]
            if "callbacks" in kwargs and "env_instantiation" in kwargs["callbacks"]
            else None
        )
        self.callbacks["env_reset"] = (
            kwargs["callbacks"]["env_reset"]
            if "callbacks" in kwargs and "env_reset" in kwargs["callbacks"]
            else None
        )
        self.callbacks["algorithm_instantiation"] = (
            kwargs["callbacks"]["algorithm_instantiation"]
            if "callbacks" in kwargs
            and "algorithm_instantiation" in kwargs["callbacks"]
            else None
        )
        self.callbacks["training_iteration"] = (
            kwargs["callbacks"]["training_iteration"]
            if "callbacks" in kwargs and "training_iteration" in kwargs["callbacks"]
            else None
        )
        self.callbacks["validation"] = (
            kwargs["callbacks"]["validation"]
            if "callbacks" in kwargs and "validation" in kwargs["callbacks"]
            else None
        )

        config_location = config_location
        parallel_expirements = parallel_expirements

        if not os.path.exists(self._expirement_save_location):
            os.makedirs(self._expirement_save_location)

        if os.path.isfile(config_location):
            self.run_expirement(config_location)
        elif os.path.isdir(config_location):
            config_files = os.listdir(config_location)

            with Pool(parallel_expirements) as pool:
                arguments = [
                    os.path.join(config_location, config_file)
                    for config_file in config_files
                ]
                self._run_expirement(arguments[0])
        else:
            logger.critical("Invalid configuration directory was provided")
            raise ValueError("Invalid directory of file given")

        logger.info(
            f"All expirements ran. See {self._expirement_save_location} for logs."
        )

    def _run_expirement(self, config_file: str) -> None:
        """
        Called for each expirement file, parses config file, spawns proc's for each
        trial. Aggregates final expirement results and saves them accordingly.

        :param config_file: location of config file for expirement
        """
        try:
            with open(config_file, "r") as json_file:
                config = json.load(json_file)
        except FileNotFoundError:
            logger.warning(f"File not found: {config_file}, moving to next file...")
            return

        if len(config["seeds"]) != config["number_of_trials"]:
            logger.error(
                f"There are more trials then given seeds in {config_file}, proceeding to next file..."
            )
            return

        validation_rewards = []
        validation_timesteps = []
        training_rewards = []
        training_timesteps = []

        for seed in config["seeds"]:
            val_rewards, val_timesteps, train_rewards, train_timesteps = (
                self._run_trial(seed, **config)
            )
            validation_rewards.append(val_rewards)
            validation_timesteps.append(val_timesteps)
            training_rewards.append(train_rewards)
            training_timesteps.append(train_timesteps)

        np.savez(
            os.path.join(
                self._expirement_save_location,
                f"{config['name']}/aggregate_results.npy",
            ),
            validation_rewards,
            validation_timesteps,
            training_rewards,
            training_timesteps,
        )
        logger.info(f"Expirement: {config['name']} complete.")

    def _run_trial(self, seed: int, **config) -> list:
        """
        Run instance of trial in expirement.

        :param config: dict parsed from config file
        :param seed: seed for random generator of trial
        """
        try:
            assert issubclass(self._env, gym.Env)

            random_number_generator = np.random.default_rng(seed)

            env = (
                self._env()
                if self.callbacks["env_instantiation"] is None
                else self.callbacks["env_instantiation"](
                    seed, random_number_generator, **config["env"]
                )
            )

            algorithm = (
                algorithms.algorithms[config["algorithm"]["name"]](
                    random_number_generator, logger, env, **config["algorithm"]
                )
                if self.callbacks["algorithm_instantiation"] is None
                else self.callbacks["algorithm_instantiation"](
                    seed, random_number_generator, **config["algorithm"]
                )
            )

            validation_rewards = []
            validation_timesteps = []

            training_rewards = []
            training_timesteps = []

            total_timestep_counter = 0

            validation_frequency = int(config["validation_frequency"])
            validation_duration = int(config["validation_duration"])
            max_episodes = int(config["max_episodes"])
            stop_reward = int(config["stop_reward"])

            for episode in range(1, max_episodes + 1):
                print("working...")
                state = (
                    env.reset()[0]
                    if self.callbacks["env_reset"] is None
                    else self.callbacks["env_reset"](
                        env, seed, random_number_generator, **config["env"]
                    )[0]
                )

                # Validation ran at provided interval
                if episode % validation_frequency == 0:
                    for _ in validation_duration:
                        validation_instance_rewards = []
                        validation_instance_timesteps = []

                        episode_rewards, episode_timesteps, total_timestep_counter = (
                            algorithm.run_episode(
                                validation=True,
                                initial_state=state,
                                timestep_counter=total_timestep_counter,
                            )
                        )
                        validation_instance_rewards.append(episode_rewards)
                        validation_instance_timesteps.append(episode_timesteps)

                    validation_rewards.append(np.average(validation_instance_rewards))
                    validation_timesteps.append(
                        np.average(validation_instance_timesteps)
                    )

                    if validation_rewards[-1] >= stop_reward:
                        break

                    if self.callbacks["validation"] is not None:
                        if self.callbacks["validation"](
                            seed,
                            random_number_generator,
                            config,
                            validation_rewards,
                            validation_timesteps,
                        ):
                            logger.info(
                                f"Expirement {config['name']}, trial {'seed'} completed from validation stop callback."
                            )
                            break

                else:
                    episode_rewards, episode_timesteps, total_timestep_counter = (
                        algorithm.run_episode(
                            validation=False,
                            initial_state=state,
                            total_timestep_counter=total_timestep_counter,
                        )
                    )
                    print(episode_timesteps)
                    print(episode_rewards)

                    training_rewards.append(episode_rewards)
                    training_timesteps.append(episode_timesteps)
                    if self.callbacks["training_iteration"] is not None:
                        if self.callbacks["training_iteration"](
                            seed,
                            random_number_generator,
                            config,
                            training_rewards,
                            validation_rewards,
                        ):
                            logger.info(
                                f"Expirement {config['name']}, trial {'seed'} completed from expirement stop callback."
                            )
                            break

            env.close()

            os.mkdir(os.path.join(self._expirement_save_location, f"{config['name']}"))

            algorithm.save_model(
                os.path.join(
                    self._expirement_save_location, f"{config['name']}/{seed}_model"
                )
            )

            np.savez(
                os.path.join(
                    self._expirement_save_location,
                    f"{config['name']}/{seed}_results.npy",
                ),
                validation_rewards,
                validation_timesteps,
                training_rewards,
                training_timesteps,
            )

            logger.info(f"{config['name']} completed.")
            return (
                validation_rewards,
                validation_timesteps,
                training_rewards,
                training_timesteps,
            )
        except KeyError as err:
            self.logger.error(
                f"Invalid config file: {config['name']}, proceeding to next file: {err}"
            )
            return
        except OSError as err:
            logger.error(f"A file issue occured, {err}")
        except AssertionError as err:
            logger.error(f"Invalid environment, is not gym env: {err}")
