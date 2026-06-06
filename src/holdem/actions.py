from enum import Enum


class Action(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


ACTION_SPACE = [
    Action.FOLD,
    Action.CHECK,
    Action.CALL,
    Action.BET,
    Action.RAISE,
]