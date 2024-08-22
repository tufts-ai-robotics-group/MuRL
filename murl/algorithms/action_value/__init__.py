from rl.algorithms.action_value.policies.greedy import Greedy
from rl.algorithms.action_value.policies.epsilon_greedy import EpsilonGreedy

from rl.algorithms.action_value.algorithms.q_learning import QLearning

action_value_policies = {
    "greedy": Greedy,
    "epsilon-greedy": EpsilonGreedy,
}

action_value_algorithms = {"q_learning": QLearning}
