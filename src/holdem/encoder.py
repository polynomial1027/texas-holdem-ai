import numpy as np

from src.holdem.actions import Action
from src.holdem.features import encode_poker_features, get_poker_feature_dim


ACTION_TO_INDEX = {
    Action.FOLD.value: 0,
    Action.CHECK.value: 1,
    Action.CALL.value: 2,
    Action.BET.value: 3,
    Action.RAISE.value: 4,
}


STREET_TO_INDEX = {
    "preflop": 0,
    "flop": 1,
    "turn": 2,
    "river": 3,
    "showdown": 4,
}


def card_to_index(card) -> int:
    """
    把一张牌编码成 0-51 的整数。

    rank:
        2 -> 0
        3 -> 1
        ...
        A -> 12

    suit:
        ♠ -> 0
        ♥ -> 1
        ♦ -> 2
        ♣ -> 3

    index = suit_index * 13 + rank_index
    """
    suit_to_index = {
        "♠": 0,
        "♥": 1,
        "♦": 2,
        "♣": 3,
    }

    rank_index = card.rank - 2
    suit_index = suit_to_index[card.suit]

    return suit_index * 13 + rank_index


def encode_cards(cards) -> np.ndarray:
    """
    把若干张牌编码成 52 维 one-hot 向量。
    """
    vector = np.zeros(52, dtype=np.float32)

    for card in cards:
        index = card_to_index(card)
        vector[index] = 1.0

    return vector


def encode_street(street: str) -> np.ndarray:
    """
    把游戏阶段编码成 5 维 one-hot。

    preflop / flop / turn / river / showdown
    """
    vector = np.zeros(5, dtype=np.float32)

    if street not in STREET_TO_INDEX:
        raise ValueError(f"Unknown street: {street}")

    vector[STREET_TO_INDEX[street]] = 1.0
    return vector


def encode_legal_actions(legal_actions) -> np.ndarray:
    """
    把合法动作编码成 5 维 action mask。

    顺序：
    0 fold
    1 check
    2 call
    3 bet
    4 raise
    """
    vector = np.zeros(5, dtype=np.float32)

    for action in legal_actions:
        if action not in ACTION_TO_INDEX:
            raise ValueError(f"Unknown action: {action}")
        vector[ACTION_TO_INDEX[action]] = 1.0

    return vector


def normalize_scalar(value: float, scale: float) -> float:
    """
    简单归一化，避免数值太大。
    """
    return float(value) / float(scale)


def encode_observation(observation) -> np.ndarray:
    """
    把 game.get_observation() 返回的 dict 编码成一个一维 numpy 向量。

    当前向量结构：

    52维：当前玩家手牌
    52维：公共牌
    5维 ：street one-hot
    1维 ：pot / 1000
    1维 ：current_bet / 1000
    2维 ：双方筹码 / 1000
    2维 ：双方当前下注 / 1000
    2维 ：双方是否 fold
    5维 ：legal action mask
    31维：poker semantic features，包括当前牌型、对子结构、同花听牌、顺子听牌、高牌潜力等

    总维度：
    122 + 31 = 153
    """
    hole_cards = observation["hole_cards"]
    board_cards = observation["board"]

    hole_cards_vec = encode_cards(hole_cards)
    board_vec = encode_cards(board_cards)
    street_vec = encode_street(observation["street"])

    pot_vec = np.array(
        [normalize_scalar(observation["pot"], 1000)],
        dtype=np.float32,
    )

    current_bet_vec = np.array(
        [normalize_scalar(observation["current_bet"], 1000)],
        dtype=np.float32,
    )

    chips_vec = np.array(
        [normalize_scalar(chips, 1000) for chips in observation["player_chips"]],
        dtype=np.float32,
    )

    current_bets_vec = np.array(
        [normalize_scalar(bet, 1000) for bet in observation["player_current_bets"]],
        dtype=np.float32,
    )

    folded_vec = np.array(
        [1.0 if folded else 0.0 for folded in observation["player_folded"]],
        dtype=np.float32,
    )

    legal_actions_vec = encode_legal_actions(observation["legal_actions"])

    poker_features_vec = encode_poker_features(
        hole_cards=hole_cards,
        board_cards=board_cards,
    )

    state = np.concatenate([
        hole_cards_vec,
        board_vec,
        street_vec,
        pot_vec,
        current_bet_vec,
        chips_vec,
        current_bets_vec,
        folded_vec,
        legal_actions_vec,
        poker_features_vec,
    ])

    return state.astype(np.float32)


def get_state_dim() -> int:
    """
    返回当前状态向量维度。
    """
    return 122 + get_poker_feature_dim()


def get_action_dim() -> int:
    """
    返回动作空间大小。
    """
    return 5