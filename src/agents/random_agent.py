import random


class RandomAgent:
    """
    最基础的随机AI。
    目前第一阶段还没有下注系统，所以这个类先作为占位。
    后面我们会让它随机选择 fold / call / raise。
    """

    def choose_action(self, legal_actions):
        if not legal_actions:
            raise ValueError("No legal actions available.")
        return random.choice(legal_actions)