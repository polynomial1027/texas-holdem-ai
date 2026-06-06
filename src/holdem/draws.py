from collections import Counter

import numpy as np

from src.holdem.features import extract_poker_feature_dict


DRAW_FEATURE_DIM = 18


def _rank_set_with_wheel(cards):
    ranks = {card.rank for card in cards}

    if 14 in ranks:
        ranks.add(1)

    return ranks


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


def _card_name(card):
    names = {
        11: "J",
        12: "Q",
        13: "K",
        14: "A",
    }

    return f"{names.get(card.rank, str(card.rank))}{card.suit}"


def analyze_flush_draw(hole_cards, board_cards):
    cards = hole_cards + board_cards
    suit_counts = Counter(card.suit for card in cards)

    best_suit = None
    best_count = 0

    for suit, count in suit_counts.items():
        if count > best_count:
            best_suit = suit
            best_count = count

    has_made_flush = best_count >= 5
    has_flush_draw = best_count == 4 and not has_made_flush
    has_backdoor_flush_draw = len(board_cards) == 3 and best_count == 3

    outs = 0
    quality = "none"
    is_nut_flush_draw = False
    is_low_flush_draw = False

    if has_flush_draw and best_suit is not None:
        outs = 13 - best_count

        hole_suited_cards = [
            card for card in hole_cards
            if card.suit == best_suit
        ]

        hole_suited_ranks = sorted(
            [card.rank for card in hole_suited_cards],
            reverse=True,
        )

        highest_hole_suited_rank = max(hole_suited_ranks, default=0)

        if highest_hole_suited_rank == 14:
            quality = "nut"
            is_nut_flush_draw = True
        elif highest_hole_suited_rank >= 11:
            quality = "high"
        elif highest_hole_suited_rank >= 8:
            quality = "medium"
        else:
            quality = "low"
            is_low_flush_draw = True

    useful_suit = best_suit if has_flush_draw else None

    return {
        "has_flush_draw": has_flush_draw,
        "has_backdoor_flush_draw": has_backdoor_flush_draw,
        "has_made_flush": has_made_flush,
        "flush_outs": outs,
        "flush_draw_quality": quality,
        "is_nut_flush_draw": is_nut_flush_draw,
        "is_low_flush_draw": is_low_flush_draw,
        "useful_flush_suit": useful_suit,
    }


def analyze_straight_draw(hole_cards, board_cards):
    cards = hole_cards + board_cards
    ranks = _rank_set_with_wheel(cards)

    made_straight = False
    open_ended_outs = set()
    gutshot_outs = set()

    for window in _straight_windows():
        hit = window & ranks
        missing = window - ranks

        if len(hit) >= 5:
            made_straight = True

        if len(hit) == 4 and len(missing) == 1:
            missing_rank = next(iter(missing))
            sorted_window = sorted(window)

            if missing_rank == sorted_window[0] or missing_rank == sorted_window[-1]:
                open_ended_outs.add(missing_rank)
            else:
                gutshot_outs.add(missing_rank)

    if made_straight:
        open_ended_outs.clear()
        gutshot_outs.clear()

    straight_out_ranks = sorted(open_ended_outs | gutshot_outs)
    straight_outs = 4 * len(straight_out_ranks)

    if open_ended_outs:
        draw_type = "open_ended"
    elif gutshot_outs:
        draw_type = "gutshot"
    else:
        draw_type = "none"

    return {
        "has_made_straight": made_straight,
        "has_open_ended_straight_draw": bool(open_ended_outs),
        "has_gutshot_straight_draw": bool(gutshot_outs),
        "has_any_straight_draw": bool(open_ended_outs or gutshot_outs),
        "straight_draw_type": draw_type,
        "straight_out_ranks": straight_out_ranks,
        "straight_outs": straight_outs,
    }


def analyze_pair_outs(hole_cards, board_cards):
    """
    粗略估算配对 outs。

    如果当前没有对子：
    - 每张未配对手牌理论上还有 3 张可以配对
    - 两张不同手牌通常最多 6 outs

    如果已经有对子或更好，则这里不再把 pair outs 当作主要目标。
    """
    cards = hole_cards + board_cards
    ranks = [card.rank for card in cards]
    counts = Counter(ranks)

    hole_ranks = [card.rank for card in hole_cards]
    has_pair_or_better = any(count >= 2 for count in counts.values())

    if has_pair_or_better:
        return {
            "pair_outs": 0,
            "target_pair": False,
            "pair_out_ranks": [],
        }

    out_ranks = sorted(set(hole_ranks))
    pair_outs = 3 * len(out_ranks)

    return {
        "pair_outs": pair_outs,
        "target_pair": pair_outs > 0,
        "pair_out_ranks": out_ranks,
    }


def analyze_draws(hole_cards, board_cards):
    """
    返回人类可读的听牌分析。

    这个函数既给 GUI 显示用，也给训练特征编码用。
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

    total_raw_outs = (
        flush_info["flush_outs"]
        + straight_info["straight_outs"]
        + pair_info["pair_outs"]
    )

    # 简化版 effective outs：
    # 低同花听牌要打折，因为 reverse implied odds 更高。
    effective_outs = total_raw_outs

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
        return "Only weak pair-improvement potential. Usually not enough for aggressive action."

    return f"Mixed weak potential with about {effective_outs} effective outs."


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

    flush_quality = info["flush_draw_quality"]

    flush_quality_vec = np.zeros(4, dtype=np.float32)

    if flush_quality == "none":
        flush_quality_vec[0] = 1.0
    elif flush_quality == "low":
        flush_quality_vec[1] = 1.0
    elif flush_quality in ["medium", "high"]:
        flush_quality_vec[2] = 1.0
    elif flush_quality == "nut":
        flush_quality_vec[3] = 1.0

    straight_type = info["straight_draw_type"]

    straight_type_vec = np.zeros(3, dtype=np.float32)

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

    if vector.shape != (DRAW_FEATURE_DIM,):
        raise RuntimeError(f"Unexpected draw feature shape: {vector.shape}")

    return vector.astype(np.float32)


def get_draw_feature_dim():
    return DRAW_FEATURE_DIM