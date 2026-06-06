from collections import Counter
from itertools import combinations

import numpy as np

from src.holdem.evaluator import evaluate_five, hand_rank_name


POKER_FEATURE_DIM = 31


HAND_RANK_TO_INDEX = {
    0: "High Card",
    1: "One Pair",
    2: "Two Pair",
    3: "Three of a Kind",
    4: "Straight",
    5: "Flush",
    6: "Full House",
    7: "Four of a Kind",
    8: "Straight Flush",
}


def _rank_counts(cards):
    return Counter(card.rank for card in cards)


def _suit_counts(cards):
    return Counter(card.suit for card in cards)


def _expanded_ranks_for_straight(cards):
    ranks = set(card.rank for card in cards)

    if 14 in ranks:
        ranks.add(1)

    return sorted(ranks)


def get_current_made_hand_rank(hole_cards, board_cards):
    """
    返回当前已成牌的等级。

    注意：
    - preflop / 少于 5 张总牌时，不能用完整 5 张牌评估。
    - 这时只根据对子结构判断。
    - flop 以后有 5 张或更多牌时，从所有 5 张组合中选最强。
    """
    cards = hole_cards + board_cards

    if len(cards) >= 5:
        best_score = None

        for five_cards in combinations(cards, 5):
            score = evaluate_five(five_cards)
            if best_score is None or score > best_score:
                best_score = score

        return best_score[0]

    counts = sorted(_rank_counts(cards).values(), reverse=True)

    if counts and counts[0] >= 4:
        return 7

    if counts and counts[0] == 3:
        return 3

    pair_count = sum(1 for count in counts if count == 2)

    if pair_count >= 2:
        return 2

    if pair_count == 1:
        return 1

    return 0


def has_flush_draw(hole_cards, board_cards):
    """
    是否有同花听牌。
    当前简化定义：任意花色已经有 4 张，但还没成同花。
    """
    cards = hole_cards + board_cards
    suit_counts = _suit_counts(cards)

    max_suit_count = max(suit_counts.values(), default=0)

    return max_suit_count == 4


def has_backdoor_flush_draw(hole_cards, board_cards):
    """
    后门同花听牌：flop 时有 3 张同花，需要 turn 和 river 都补同花。
    """
    cards = hole_cards + board_cards

    if len(board_cards) != 3:
        return False

    suit_counts = _suit_counts(cards)
    max_suit_count = max(suit_counts.values(), default=0)

    return max_suit_count == 3


def has_made_flush(hole_cards, board_cards):
    cards = hole_cards + board_cards
    suit_counts = _suit_counts(cards)
    return max(suit_counts.values(), default=0) >= 5


def detect_straight_draws(hole_cards, board_cards):
    """
    检测顺子听牌。

    返回：
    {
        "has_open_ended_straight_draw": bool,
        "has_gutshot_straight_draw": bool,
        "has_any_straight_draw": bool,
        "has_made_straight": bool,
    }

    简化思路：
    - 枚举所有 5 连窗口。
    - 如果当前已有 5 张命中，则已经成顺。
    - 如果已有 4 张命中：
        缺两端之一 => open-ended
        缺中间 => gutshot
    """
    cards = hole_cards + board_cards
    ranks = set(_expanded_ranks_for_straight(cards))

    windows = [
        {1, 2, 3, 4, 5},
        {2, 3, 4, 5, 6},
        {3, 4, 5, 6, 7},
        {4, 5, 6, 7, 8},
        {5, 6, 7, 8, 9},
        {6, 7, 8, 9, 10},
        {7, 8, 9, 10, 11},
        {8, 9, 10, 11, 12},
        {9, 10, 11, 12, 13},
        {10, 11, 12, 13, 14},
    ]

    has_made_straight = False
    has_open_ended = False
    has_gutshot = False

    for window in windows:
        hit = window & ranks
        missing = sorted(window - ranks)

        if len(hit) >= 5:
            has_made_straight = True

        if len(hit) == 4 and len(missing) == 1:
            missing_rank = missing[0]
            sorted_window = sorted(window)

            # 缺的是两端，近似认为是 open-ended
            if missing_rank == sorted_window[0] or missing_rank == sorted_window[-1]:
                has_open_ended = True
            else:
                has_gutshot = True

    return {
        "has_open_ended_straight_draw": has_open_ended and not has_made_straight,
        "has_gutshot_straight_draw": has_gutshot and not has_made_straight,
        "has_any_straight_draw": (has_open_ended or has_gutshot) and not has_made_straight,
        "has_made_straight": has_made_straight,
    }


def has_two_high_cards(hole_cards):
    """
    两张手牌是否都是 T/J/Q/K/A。
    """
    return len(hole_cards) == 2 and all(card.rank >= 10 for card in hole_cards)


def has_overcards(hole_cards, board_cards):
    """
    是否有 overcards。
    例如 KQ 在 J85 board 上，两张都高于 board 最大牌。
    """
    if not board_cards:
        return False

    board_max = max(card.rank for card in board_cards)

    return any(card.rank > board_max for card in hole_cards)


def count_overcards(hole_cards, board_cards):
    if not board_cards:
        return 0

    board_max = max(card.rank for card in board_cards)

    return sum(1 for card in hole_cards if card.rank > board_max)


def get_pair_structure_features(hole_cards, board_cards):
    cards = hole_cards + board_cards
    counts = sorted(_rank_counts(cards).values(), reverse=True)

    pair_count = sum(1 for count in counts if count == 2)
    has_pair = pair_count >= 1
    has_two_pair = pair_count >= 2
    has_trips = any(count == 3 for count in counts)
    has_quads = any(count == 4 for count in counts)

    return {
        "has_pair": has_pair,
        "has_two_pair": has_two_pair,
        "has_trips": has_trips,
        "has_quads": has_quads,
    }


def extract_poker_feature_dict(hole_cards, board_cards):
    """
    输出人类可读的扑克语义特征。
    GUI 后面也可以直接显示这个 dict。
    """
    made_hand_rank = get_current_made_hand_rank(hole_cards, board_cards)
    straight_info = detect_straight_draws(hole_cards, board_cards)
    pair_info = get_pair_structure_features(hole_cards, board_cards)

    board_count = len(board_cards)
    remaining_board_cards = max(0, 5 - board_count)

    made_hand_name = HAND_RANK_TO_INDEX[made_hand_rank]

    has_pair_or_better = made_hand_rank >= 1
    has_strong_made_hand = made_hand_rank >= 2
    has_very_strong_made_hand = made_hand_rank >= 4

    flush_draw = has_flush_draw(hole_cards, board_cards)
    backdoor_flush_draw = has_backdoor_flush_draw(hole_cards, board_cards)
    made_flush = has_made_flush(hole_cards, board_cards)

    any_draw = (
        flush_draw
        or backdoor_flush_draw
        or straight_info["has_any_straight_draw"]
    )

    air = (
        made_hand_rank == 0
        and not any_draw
        and not has_overcards(hole_cards, board_cards)
    )

    return {
        "made_hand_rank": made_hand_rank,
        "made_hand_name": made_hand_name,
        "has_pair_or_better": has_pair_or_better,
        "has_strong_made_hand": has_strong_made_hand,
        "has_very_strong_made_hand": has_very_strong_made_hand,
        "has_pair": pair_info["has_pair"],
        "has_two_pair": pair_info["has_two_pair"],
        "has_trips": pair_info["has_trips"],
        "has_quads": pair_info["has_quads"],
        "has_flush_draw": flush_draw,
        "has_backdoor_flush_draw": backdoor_flush_draw,
        "has_made_flush": made_flush,
        "has_open_ended_straight_draw": straight_info["has_open_ended_straight_draw"],
        "has_gutshot_straight_draw": straight_info["has_gutshot_straight_draw"],
        "has_any_straight_draw": straight_info["has_any_straight_draw"],
        "has_made_straight": straight_info["has_made_straight"],
        "has_two_high_cards": has_two_high_cards(hole_cards),
        "has_overcards": has_overcards(hole_cards, board_cards),
        "overcard_count": count_overcards(hole_cards, board_cards),
        "board_count": board_count,
        "remaining_board_cards": remaining_board_cards,
        "is_air": air,
        "has_any_draw": any_draw,
    }


def encode_poker_features(hole_cards, board_cards):
    """
    把扑克语义特征编码成固定长度向量。

    维度设计：
    9维：made hand rank one-hot
    1维：made hand rank / 8
    1维：has_pair_or_better
    1维：has_strong_made_hand
    1维：has_very_strong_made_hand
    1维：has_pair
    1维：has_two_pair
    1维：has_trips
    1维：has_quads
    1维：has_flush_draw
    1维：has_backdoor_flush_draw
    1维：has_made_flush
    1维：has_open_ended_straight_draw
    1维：has_gutshot_straight_draw
    1维：has_any_straight_draw
    1维：has_made_straight
    1维：has_two_high_cards
    1维：has_overcards
    1维：overcard_count / 2
    1维：board_count / 5
    1维：remaining_board_cards / 5
    1维：is_air
    1维：has_any_draw

    总维度：31
    """
    feature_dict = extract_poker_feature_dict(hole_cards, board_cards)

    made_hand_one_hot = np.zeros(9, dtype=np.float32)
    made_hand_one_hot[feature_dict["made_hand_rank"]] = 1.0

    values = [
        feature_dict["made_hand_rank"] / 8.0,
        float(feature_dict["has_pair_or_better"]),
        float(feature_dict["has_strong_made_hand"]),
        float(feature_dict["has_very_strong_made_hand"]),
        float(feature_dict["has_pair"]),
        float(feature_dict["has_two_pair"]),
        float(feature_dict["has_trips"]),
        float(feature_dict["has_quads"]),
        float(feature_dict["has_flush_draw"]),
        float(feature_dict["has_backdoor_flush_draw"]),
        float(feature_dict["has_made_flush"]),
        float(feature_dict["has_open_ended_straight_draw"]),
        float(feature_dict["has_gutshot_straight_draw"]),
        float(feature_dict["has_any_straight_draw"]),
        float(feature_dict["has_made_straight"]),
        float(feature_dict["has_two_high_cards"]),
        float(feature_dict["has_overcards"]),
        feature_dict["overcard_count"] / 2.0,
        feature_dict["board_count"] / 5.0,
        feature_dict["remaining_board_cards"] / 5.0,
        float(feature_dict["is_air"]),
        float(feature_dict["has_any_draw"]),
    ]

    feature_vector = np.concatenate([
        made_hand_one_hot,
        np.array(values, dtype=np.float32),
    ])

    if feature_vector.shape != (POKER_FEATURE_DIM,):
        raise RuntimeError(
            f"Unexpected poker feature shape: {feature_vector.shape}"
        )

    return feature_vector.astype(np.float32)


def get_poker_feature_dim():
    return POKER_FEATURE_DIM