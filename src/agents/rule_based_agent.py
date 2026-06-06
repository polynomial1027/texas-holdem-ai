from src.holdem.features import analyze_draws


class RuleBasedAgent:
    """
    一个基础规则 AI。

    目标：
    - 不像 RandomAgent 那样乱打
    - 会根据当前牌型、听牌、outs 做基础判断
    - 作为 DQN 第二阶段训练对手

    规则不是 GTO，只是一个更合理的训练对手。
    """

    def choose_action(self, observation):
        legal_actions = observation["legal_actions"]

        if not legal_actions:
            raise ValueError("No legal actions available.")

        hole_cards = observation["hole_cards"]
        board_cards = observation["board"]

        draw_info = analyze_draws(
            hole_cards=hole_cards,
            board_cards=board_cards,
        )

        facing_bet = "call" in legal_actions
        can_raise = "raise" in legal_actions
        can_bet = "bet" in legal_actions
        can_check = "check" in legal_actions
        can_fold = "fold" in legal_actions
        can_call = "call" in legal_actions

        made_hand_rank = draw_info["made_hand_rank"]
        effective_outs = draw_info["effective_outs"]

        is_effective_air = draw_info["is_effective_air"]
        has_flush_draw = draw_info["has_flush_draw"]
        has_open_ended = draw_info["has_open_ended_straight_draw"]
        has_gutshot = draw_info["has_gutshot_straight_draw"]
        is_nut_flush_draw = draw_info["is_nut_flush_draw"]
        is_low_flush_draw = draw_info["is_low_flush_draw"]

        # ------------------------------------------------------------
        # 情况一：面对对手下注
        # 合法动作通常是 fold / call / raise
        # ------------------------------------------------------------
        if facing_bet:
            # 强成牌：顺子及以上
            if made_hand_rank >= 4:
                if can_raise:
                    return "raise"
                if can_call:
                    return "call"

            # 中强成牌：两对 / 三条
            if made_hand_rank >= 2:
                if can_call:
                    return "call"

            # 顶级同花听牌：可以积极一点
            if is_nut_flush_draw:
                if can_raise:
                    return "raise"
                if can_call:
                    return "call"

            # 开放式顺子听牌：通常可以跟
            if has_open_ended:
                if can_call:
                    return "call"

            # 普通同花听牌：可以跟，但低同花听牌不轻易 raise
            if has_flush_draw:
                if is_low_flush_draw:
                    if can_call:
                        return "call"
                else:
                    if effective_outs >= 8 and can_call:
                        return "call"

            # 卡顺：弱一些，小心跟注
            if has_gutshot:
                if effective_outs >= 4 and can_call:
                    return "call"

            # 空气牌面对下注：默认弃牌
            if is_effective_air:
                if can_fold:
                    return "fold"

            # 没有强理由继续时，能弃就弃
            if can_fold:
                return "fold"

            # 兜底
            if can_call:
                return "call"

        # ------------------------------------------------------------
        # 情况二：当前没人下注
        # 合法动作通常是 check / bet
        # ------------------------------------------------------------
        else:
            # 强成牌主动下注
            if made_hand_rank >= 2:
                if can_bet:
                    return "bet"
                if can_check:
                    return "check"

            # 强听牌可以 semi-bluff bet
            if is_nut_flush_draw or has_open_ended:
                if can_bet:
                    return "bet"
                if can_check:
                    return "check"

            # 普通听牌可以 check
            if has_flush_draw or has_gutshot:
                if can_check:
                    return "check"

            # 空气牌没人下注就 check
            if can_check:
                return "check"

            if can_bet:
                return "bet"

        # 最后兜底：返回第一个合法动作
        return legal_actions[0]