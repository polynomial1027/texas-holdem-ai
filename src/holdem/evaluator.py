from collections import Counter
from itertools import combinations


HAND_RANK_NAME = {
    8: "Straight Flush",
    7: "Four of a Kind",
    6: "Full House",
    5: "Flush",
    4: "Straight",
    3: "Three of a Kind",
    2: "Two Pair",
    1: "One Pair",
    0: "High Card",
}


def _straight_high_card(ranks):
    """
    判断是否为顺子。
    返回顺子的最大牌点数。
    A2345 视为 5-high straight。
    """
    unique = sorted(set(ranks), reverse=True)

    # A 可以当作 1
    if 14 in unique:
        unique.append(1)

    for i in range(len(unique) - 4):
        window = unique[i:i + 5]
        if window[0] - window[4] == 4 and len(set(window)) == 5:
            return window[0]

    return None


def evaluate_five(cards):
    """
    评估5张牌。
    
    返回一个 tuple，越大越强。
    格式类似：
    (牌型等级, 关键牌1, 关键牌2, ...)
    """
    ranks = [card.rank for card in cards]
    suits = [card.suit for card in cards]

    rank_counts = Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)

    is_flush = len(set(suits)) == 1
    straight_high = _straight_high_card(ranks)

    # 同花顺
    if is_flush and straight_high:
        return (8, straight_high)

    # 四条
    if counts == [4, 1]:
        four_rank = max(rank for rank, count in rank_counts.items() if count == 4)
        kicker = max(rank for rank, count in rank_counts.items() if count == 1)
        return (7, four_rank, kicker)

    # 葫芦
    if counts == [3, 2]:
        three_rank = max(rank for rank, count in rank_counts.items() if count == 3)
        pair_rank = max(rank for rank, count in rank_counts.items() if count == 2)
        return (6, three_rank, pair_rank)

    # 同花
    if is_flush:
        return (5, *sorted(ranks, reverse=True))

    # 顺子
    if straight_high:
        return (4, straight_high)

    # 三条
    if counts == [3, 1, 1]:
        three_rank = max(rank for rank, count in rank_counts.items() if count == 3)
        kickers = sorted(
            [rank for rank, count in rank_counts.items() if count == 1],
            reverse=True
        )
        return (3, three_rank, *kickers)

    # 两对
    if counts == [2, 2, 1]:
        pairs = sorted(
            [rank for rank, count in rank_counts.items() if count == 2],
            reverse=True
        )
        kicker = max(rank for rank, count in rank_counts.items() if count == 1)
        return (2, *pairs, kicker)

    # 一对
    if counts == [2, 1, 1, 1]:
        pair_rank = max(rank for rank, count in rank_counts.items() if count == 2)
        kickers = sorted(
            [rank for rank, count in rank_counts.items() if count == 1],
            reverse=True
        )
        return (1, pair_rank, *kickers)

    # 高牌
    return (0, *sorted(ranks, reverse=True))


def evaluate_seven(cards):
    """
    从7张牌中选择最强的5张牌。
    """
    if len(cards) != 7:
        raise ValueError("Texas Hold'em hand evaluation requires exactly 7 cards.")

    best_score = None
    best_five = None

    for five_cards in combinations(cards, 5):
        score = evaluate_five(five_cards)
        if best_score is None or score > best_score:
            best_score = score
            best_five = five_cards

    return best_score, list(best_five)


def hand_rank_name(score):
    return HAND_RANK_NAME[score[0]]