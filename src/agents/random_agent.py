import random


class RandomAgent:
    """
    随机策略 AI。
    它只会从合法动作里随机选一个。
    """

    def choose_action(self, observation):
        legal_actions = observation["legal_actions"]

        if not legal_actions:
            raise ValueError("No legal actions available.")

        return random.choice(legal_actions)