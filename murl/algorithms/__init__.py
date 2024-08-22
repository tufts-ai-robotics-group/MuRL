import rl.algorithms.action_value as action_value
import rl.algorithms.policy_gradient as policy_gradient

algorithms = {}

algorithms.update(action_value.action_value_algorithms)
algorithms.update(policy_gradient.policy_gradient_algorithms)
