from collections import Counter
from itertools import combinations

import numpy as np

from src.holdem.evaluator import evaluate_five


# 31 basic poker semantic features + 18 draw/outs features
POKER_FEATURE_DIM = 49


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


def _straight_windows():
    return [
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


def _rank_name(rank):
    names = {
        1: "A",
        11: "J",
        12: "Q",
        13: "K",
        14: "A",
    }
    return names.get(rank, str(rank))


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
    """
    cards = hole_cards + board_cards
    ranks = set(_expanded_ranks_for_straight(cards))

    has_made_straight = False
    has_open_ended = False
    has_gutshot = False

    for window in _straight_windows():
        hit = window & ranks
        missing = sorted(window - ranks)

        if len(hit) >= 5:
            has_made_straight = True

        if len(hit) == 4 and len(missing) == 1:
            missing_rank = missing[0]
            sorted_window = sorted(window)

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


def analyze_flush_draw(hole_cards, board_cards):
    """
    分析同花听牌、同花 outs 和同花听牌质量。

    quality:
    - none: 没有同花听牌
    - low: 低同花听牌，例如 2♦3♦ 在 A♦K♦Q♥ 上
    - medium: 中等同花听牌
    - high: 高同花听牌
    - nut: A-high nut flush draw
    """
    cards = hole_cards + board_cards
    suit_counts = _suit_counts(cards)

    best_suit = None
    best_count = 0

    for suit, count in suit_counts.items():
        if count > best_count:
            best_suit = suit
            best_count = count

    made_flush = best_count >= 5
    flush_draw = best_count == 4 and not made_flush
    backdoor_flush_draw = len(board_cards) == 3 and best_count == 3

    outs = 0
    quality = "none"
    is_nut_flush_draw = False
    is_low_flush_draw = False

    if flush_draw and best_suit is not None:
        outs = 13 - best_count

        suited_hole_cards = [card for card in hole_cards if card.suit == best_suit]
        suited_hole_ranks = [card.rank for card in suited_hole_cards]
        highest_suited_hole_rank = max(suited_hole_ranks, default=0)

        if highest_suited_hole_rank == 14:
            quality = "nut"
            is_nut_flush_draw = True
        elif highest_suited_hole_rank >= 12:
            quality = "high"
        elif highest_suited_hole_rank >= 8:
            quality = "medium"
        else:
            quality = "low"
            is_low_flush_draw = True

    return {
        "has_flush_draw": flush_draw,
        "has_backdoor_flush_draw": backdoor_flush_draw,
        "has_made_flush": made_flush,
        "flush_outs": outs,
        "flush_draw_quality": quality,
        "is_nut_flush_draw": is_nut_flush_draw,
        "is_low_flush_draw": is_low_flush_draw,
        "useful_flush_suit": best_suit if flush_draw else None,
    }


def analyze_straight_draw(hole_cards, board_cards):
    """
    分析顺子听牌和顺子 outs。
    """
    cards = hole_cards + board_cards
    ranks = set(_expanded_ranks_for_straight(cards))

    made_straight = False
    open_ended_out_ranks = set()
    gutshot_out_ranks = set()

    for window in _straight_windows():
        hit = window & ranks
        missing = window - ranks

        if len(hit) >= 5:
            made_straight = True

        if len(hit) == 4 and len(missing) == 1:
            missing_rank = next(iter(missing))
            sorted_window = sorted(window)

            if missing_rank == sorted_window[0] or missing_rank == sorted_window[-1]:
                open_ended_out_ranks.add(missing_rank)
            else:
                gutshot_out_ranks.add(missing_rank)

    if made_straight:
        open_ended_out_ranks.clear()
        gutshot_out_ranks.clear()

    straight_out_ranks = sorted(open_ended_out_ranks | gutshot_out_ranks)
    straight_outs = 4 * len(straight_out_ranks)

    if open_ended_out_ranks:
        draw_type = "open_ended"
    elif gutshot_out_ranks:
        draw_type = "gutshot"
    else:
        draw_type = "none"

    return {
        "has_made_straight": made_straight,
        "has_open_ended_straight_draw": bool(open_ended_out_ranks),
        "has_gutshot_straight_draw": bool(gutshot_out_ranks),
        "has_any_straight_draw": bool(open_ended_out_ranks or gutshot_out_ranks),
        "straight_draw_type": draw_type,
        "straight_out_ranks": straight_out_ranks,
        "straight_outs": straight_outs,
    }


def analyze_pair_outs(hole_cards, board_cards):
    """
    估算配对 outs，并区分有效配对 outs 和弱配对 outs。

    关键思想：
    - 低牌在高牌公共牌面上配成一对，通常不算强目标。
      例如 2♠3♥ / A♦Q♣9♠，配到 2 或 3 仍然很弱。
    - 高牌 overcards 配对更有价值。
      例如 KQ / J85，K 或 Q 配对更有意义。
    """
    cards = hole_cards + board_cards
    counts = _rank_counts(cards)
    hole_ranks = [card.rank for card in hole_cards]
    has_pair_or_better = any(count >= 2 for count in counts.values())

    if has_pair_or_better:
        return {
            "pair_outs": 0,
            "weak_pair_outs": 0,
            "target_pair": False,
            "pair_out_ranks": [],
            "weak_pair_out_ranks": [],
            "pair_out_quality": "none",
        }

    if not board_cards:
        out_ranks = sorted(set(hole_ranks))
        return {
            "pair_outs": 3 * len(out_ranks),
            "weak_pair_outs": 0,
            "target_pair": True,
            "pair_out_ranks": out_ranks,
            "weak_pair_out_ranks": [],
            "pair_out_quality": "preflop",
        }

    board_max = max(card.rank for card in board_cards)

    strong_out_ranks = []
    weak_out_ranks = []

    for rank in sorted(set(hole_ranks)):
        if rank >= 10 or rank > board_max:
            strong_out_ranks.append(rank)
        else:
            weak_out_ranks.append(rank)

    pair_outs = 3 * len(strong_out_ranks)
    weak_pair_outs = 3 * len(weak_out_ranks)

    if strong_out_ranks:
        quality = "strong"
    elif weak_out_ranks:
        quality = "weak"
    else:
        quality = "none"

    return {
        "pair_outs": pair_outs,
        "weak_pair_outs": weak_pair_outs,
        "target_pair": pair_outs > 0,
        "pair_out_ranks": strong_out_ranks,
        "weak_pair_out_ranks": weak_out_ranks,
        "pair_out_quality": quality,
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


def build_useful_cards_text(flush_info, straight_info, pair_info):
    useful = []

    if flush_info["has_flush_draw"] and flush_info["useful_flush_suit"]:
        useful.append(f"Any {flush_info['useful_flush_suit']}")

    if straight_info["straight_out_ranks"]:
        ranks = ", ".join(_rank_name(rank) for rank in straight_info["straight_out_ranks"])
        useful.append(f"Straight outs: {ranks}")

    if pair_info["target_pair"] and pair_info["pair_out_ranks"]:
        ranks = ", ".join(_rank_name(rank) for rank in pair_info["pair_out_ranks"])
        useful.append(f"Pair outs: {ranks}")
    elif pair_info.get("weak_pair_outs", 0) > 0:
        weak_ranks = ", ".join(
            _rank_name(rank) for rank in pair_info.get("weak_pair_out_ranks", [])
        )
        useful.append(f"Only weak low-pair improvement cards: {weak_ranks}")

    if not useful:
        return "No clear useful target cards."

    return "; ".join(useful)


def build_interpretation(
    poker_features,
    flush_info,
    straight_info,
    pair_info,
    effective_outs,
    is_effective_air,
):
    if poker_features["made_hand_rank"] >= 4:
        return "Strong made hand. Value betting or raising can be reasonable."

    if poker_features["has_strong_made_hand"]:
        return "Made hand with real showdown value. Continue carefully depending on bet size."

    if flush_info["is_nut_flush_draw"]:
        return "Nut flush draw. Strong drawing hand; aggressive semi-bluff can be reasonable."

    if flush_info["is_low_flush_draw"]:
        return "Low flush draw. It has improvement potential, but raising is risky because higher flushes can dominate."

    if straight_info["has_open_ended_straight_draw"]:
        return "Open-ended straight draw. Good drawing potential."

    if straight_info["has_gutshot_straight_draw"]:
        return "Gutshot straight draw. Some potential, but weaker than open-ended draws."

    if is_effective_air:
        return "Mostly air: no made hand, no effective draw, and no overcards. Facing a bet, folding is usually safer."

    if poker_features["has_overcards"]:
        return "High-card hand with overcards. It can improve to top pair, but should not be overplayed."

    if pair_info["target_pair"]:
        return "Pair-improvement potential. Continue cautiously unless the pair would be strong."

    if pair_info.get("weak_pair_outs", 0) > 0:
        return "Mostly air. Pairing low hole cards would still make a weak pair, so aggressive action is risky."

    return f"Mixed weak potential with about {effective_outs} effective outs."


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


def analyze_draws(hole_cards, board_cards):
    """
    返回当前牌面正在期待什么牌型、哪些牌有帮助、outs 数量和解释。
    这个函数既可以给 GUI 显示，也可以用于训练特征。
    """
    poker_features = extract_poker_feature_dict(hole_cards, board_cards)
    flush_info = analyze_flush_draw(hole_cards, board_cards)
    straight_info = analyze_straight_draw(hole_cards, board_cards)
    pair_info = analyze_pair_outs(hole_cards, board_cards)

    target_hands = []
    draw_types = []

    if flush_info["has_flush_draw"]:
        target_hands.append("Flush")
        draw_types.append(f"{flush_info['flush_draw_quality'].title()} Flush Draw")

    if straight_info["has_open_ended_straight_draw"]:
        target_hands.append("Straight")
        draw_types.append("Open-ended Straight Draw")
    elif straight_info["has_gutshot_straight_draw"]:
        target_hands.append("Straight")
        draw_types.append("Gutshot Straight Draw")

    if pair_info["target_pair"]:
        target_hands.append("Pair")
        draw_types.append("Pair Outs")
    elif pair_info.get("weak_pair_outs", 0) > 0:
        draw_types.append("Weak Pair Outs")

    total_raw_outs = (
        flush_info["flush_outs"]
        + straight_info["straight_outs"]
        + pair_info["pair_outs"]
    )

    effective_outs = total_raw_outs

    # 低同花听牌要打折，因为成牌后仍可能被更高同花压制。
    if flush_info["is_low_flush_draw"]:
        effective_outs -= 3

    effective_outs = max(0, effective_outs)

    is_effective_air = (
        poker_features["made_hand_rank"] == 0
        and not flush_info["has_flush_draw"]
        and not straight_info["has_any_straight_draw"]
        and not poker_features["has_overcards"]
    )

    if not target_hands:
        target_hands.append("None")

    if not draw_types:
        draw_types.append("No meaningful draw")

    interpretation = build_interpretation(
        poker_features=poker_features,
        flush_info=flush_info,
        straight_info=straight_info,
        pair_info=pair_info,
        effective_outs=effective_outs,
        is_effective_air=is_effective_air,
    )

    useful_cards = build_useful_cards_text(
        flush_info=flush_info,
        straight_info=straight_info,
        pair_info=pair_info,
    )

    return {
        "current_hand": poker_features["made_hand_name"],
        "made_hand_rank": poker_features["made_hand_rank"],
        "target_hands": target_hands,
        "draw_types": draw_types,
        "useful_cards": useful_cards,
        "flush_outs": flush_info["flush_outs"],
        "straight_outs": straight_info["straight_outs"],
        "pair_outs": pair_info["pair_outs"],
        "weak_pair_outs": pair_info.get("weak_pair_outs", 0),
        "pair_out_quality": pair_info.get("pair_out_quality", "none"),
        "total_raw_outs": total_raw_outs,
        "effective_outs": effective_outs,
        "flush_draw_quality": flush_info["flush_draw_quality"],
        "straight_draw_type": straight_info["straight_draw_type"],
        "is_nut_flush_draw": flush_info["is_nut_flush_draw"],
        "is_low_flush_draw": flush_info["is_low_flush_draw"],
        "has_flush_draw": flush_info["has_flush_draw"],
        "has_open_ended_straight_draw": straight_info["has_open_ended_straight_draw"],
        "has_gutshot_straight_draw": straight_info["has_gutshot_straight_draw"],
        "has_any_straight_draw": straight_info["has_any_straight_draw"],
        "has_overcards": poker_features["has_overcards"],
        "is_effective_air": is_effective_air,
        "interpretation": interpretation,
    }


def encode_draw_features(hole_cards, board_cards):
    """
    把 draw analysis 编码成训练向量。

    维度：
    1 effective_outs / 15
    1 total_raw_outs / 20
    1 flush_outs / 9
    1 straight_outs / 8
    1 pair_outs / 6
    4 flush quality one-hot: none, low, medium/high, nut
    3 straight type one-hot: none, gutshot, open_ended
    1 is_nut_flush_draw
    1 is_low_flush_draw
    1 is_effective_air
    1 target_flush
    1 target_straight
    1 target_pair

    总维度：18
    """
    info = analyze_draws(hole_cards, board_cards)

    flush_quality_vec = np.zeros(4, dtype=np.float32)
    flush_quality = info["flush_draw_quality"]

    if flush_quality == "none":
        flush_quality_vec[0] = 1.0
    elif flush_quality == "low":
        flush_quality_vec[1] = 1.0
    elif flush_quality in ["medium", "high"]:
        flush_quality_vec[2] = 1.0
    elif flush_quality == "nut":
        flush_quality_vec[3] = 1.0

    straight_type_vec = np.zeros(3, dtype=np.float32)
    straight_type = info["straight_draw_type"]

    if straight_type == "none":
        straight_type_vec[0] = 1.0
    elif straight_type == "gutshot":
        straight_type_vec[1] = 1.0
    elif straight_type == "open_ended":
        straight_type_vec[2] = 1.0

    scalar_values = np.array(
        [
            info["effective_outs"] / 15.0,
            info["total_raw_outs"] / 20.0,
            info["flush_outs"] / 9.0,
            info["straight_outs"] / 8.0,
            info["pair_outs"] / 6.0,
            float(info["is_nut_flush_draw"]),
            float(info["is_low_flush_draw"]),
            float(info["is_effective_air"]),
            float(info["has_flush_draw"]),
            float(info["has_any_straight_draw"]),
            float(info["pair_outs"] > 0),
        ],
        dtype=np.float32,
    )

    vector = np.concatenate([
        scalar_values[:5],
        flush_quality_vec,
        straight_type_vec,
        scalar_values[5:],
    ])

    if vector.shape != (18,):
        raise RuntimeError(f"Unexpected draw feature shape: {vector.shape}")

    return vector.astype(np.float32)


def encode_poker_features(hole_cards, board_cards):
    """
    把扑克语义特征编码成固定长度向量。

    结构：
    - 31维基础牌力语义特征
    - 18维听牌 / outs / 目标牌型特征

    总维度：49
    """
    feature_dict = extract_poker_feature_dict(hole_cards, board_cards)

    made_hand_one_hot = np.zeros(9, dtype=np.float32)
    made_hand_one_hot[feature_dict["made_hand_rank"]] = 1.0

    basic_values = [
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

    basic_feature_vector = np.concatenate([
        made_hand_one_hot,
        np.array(basic_values, dtype=np.float32),
    ])

    draw_feature_vector = encode_draw_features(hole_cards, board_cards)

    feature_vector = np.concatenate([
        basic_feature_vector,
        draw_feature_vector,
    ])

    if feature_vector.shape != (POKER_FEATURE_DIM,):
        raise RuntimeError(
            f"Unexpected poker feature shape: {feature_vector.shape}"
        )

    return feature_vector.astype(np.float32)


def get_poker_feature_dim():
    return POKER_FEATURE_DIM