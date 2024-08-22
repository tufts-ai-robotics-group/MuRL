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

import rl.algorithms as algorithms


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
        self._expirement_save_location = expirement_save_location

        # Callback system
        self._env_instantiation_callback = (
            kwargs["env_instantiation_callback"]
            if "env_instantiation_callback" in kwargs
            else None
        )
        self._env_reset_callback = (
            kwargs["env_reset_callback"]
            if "env_instantiation_callback" in kwargs
            else None
        )
        self._algorithm_instantiation_callback = (
            kwargs["algorithm_instantiation_callback"]
            if "algorithm_instantiation_callback" in kwargs
            else None
        )
        self._stop_training_callback = (
            kwargs["stop_training_callback"]
            if "stop_training_callback" in kwargs
            else None
        )
        self._stop_validation_callback = (
            kwargs["stop_validation_callback"]
            if "stop_validation_callback" in kwargs
            else None
        )

        self.logger = logging.getLogger(__name__)

        config_location = config_location
        parallel_expirements = parallel_expirements

        if not os.path.exists(self._expirement_save_location):
            os.makedirs(self._expirement_save_location)

        logging.basicConfig(
            filename=os.path.join(self._expirement_save_location, "expirement.log"),
            encoding="utf-8",
            level=logging.DEBUG,
        )
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
            raise ValueError("Invalid directory of file given")

        self.logger.info(
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
            self.logger.warning(
                f"File not found: {config_file}, moving to next file..."
            )
            return

        if len(config["seeds"]) != config["number_of_trials"]:
            self.logger.error(
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
        self.logger.info(f"Expirement: {config['name']} complete.")

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
                if self._env_instantiation_callback is None
                else self._env_instantiation_callback(
                    seed, random_number_generator, **config["env"]
                )
            )

            algorithm = (
                algorithms.algorithms[config["algorithm"]["name"]](
                    random_number_generator, self.logger, env, **config["algorithm"]
                )
                if self._algorithm_instantiation_callback is None
                else self._algorithm_instantiation_callback(
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
                    if self._env_reset_callback is None
                    else self._env_reset_callback(
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

                    if self._stop_validation_callback is not None:
                        if self._stop_validation_callback(
                            seed,
                            random_number_generator,
                            config,
                            validation_rewards,
                            validation_timesteps,
                        ):
                            self.logger.info(
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
                    if self._stop_training_callback is not None:
                        if self._stop_training_callback(
                            seed,
                            random_number_generator,
                            config,
                            training_rewards,
                            validation_rewards,
                        ):
                            self.logger.info(
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

            self.logger.info(f"{config['name']} completed.")
            return (
                validation_rewards,
                validation_timesteps,
                training_rewards,
                training_timesteps,
            )
        exc
        except KeyError as err:
            self.logger.error(
                f"Invalid config file: {config['name']}, proceeding to next file: {err}"
            )
            return
        except OSError as err:
            self.logger.error(f"A file issue occured, {err}")
        except AssertionError as err:
            self.logger.error(f"Invalid environment, is not gym env: {err}")
