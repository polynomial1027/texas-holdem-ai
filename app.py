import streamlit as st

from src.holdem.features import analyze_draws
from src.inference.predict_action import predict_action


RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
SUITS = {
    "♠": "s",
    "♥": "h",
    "♦": "d",
    "♣": "c",
}

ACTION_LABELS = {
    "fold": "Fold 弃牌",
    "check": "Check 过牌",
    "call": "Call 跟注",
    "bet": "Bet 下注",
    "raise": "Raise 加注",
}

EXPECTED_BOARD_COUNT = {
    "preflop": 0,
    "flop": 3,
    "turn": 4,
    "river": 5,
}


def get_expected_current_bet(my_current_bet, opponent_current_bet):
    return max(int(my_current_bet), int(opponent_current_bet))


def inject_css():
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #666;
            font-size: 1.05rem;
            margin-bottom: 1.2rem;
        }
        .card-box {
            border: 1px solid rgba(120, 120, 120, 0.25);
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            background: rgba(250, 250, 250, 0.55);
            margin-bottom: 1rem;
        }
        .result-box {
            border-radius: 20px;
            padding: 1.3rem 1.5rem;
            background: linear-gradient(135deg, rgba(20, 120, 80, 0.12), rgba(60, 120, 220, 0.10));
            border: 1px solid rgba(20, 120, 80, 0.20);
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .recommended-action {
            font-size: 2.2rem;
            font-weight: 900;
            margin: 0.2rem 0;
        }
        .small-muted {
            color: #777;
            font-size: 0.92rem;
        }
        .action-row {
            border: 1px solid rgba(120, 120, 120, 0.20);
            border-radius: 14px;
            padding: 0.7rem 0.9rem;
            margin-bottom: 0.55rem;
            background: rgba(255, 255, 255, 0.55);
        }
        .action-row-strong {
            border: 2px solid rgba(20, 140, 90, 0.65);
            background: rgba(20, 140, 90, 0.10);
        }
        .card-pill {
            display: inline-block;
            padding: 0.25rem 0.55rem;
            margin: 0.15rem 0.2rem 0.15rem 0;
            border-radius: 999px;
            border: 1px solid rgba(120, 120, 120, 0.35);
            background: rgba(255, 255, 255, 0.70);
            font-weight: 700;
        }
        .analysis-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.65rem;
            margin-top: 0.5rem;
        }
        .analysis-item {
            border: 1px solid rgba(120, 120, 120, 0.20);
            border-radius: 14px;
            padding: 0.75rem 0.85rem;
            background: rgba(255, 255, 255, 0.55);
        }
        .analysis-label {
            color: #777;
            font-size: 0.82rem;
            margin-bottom: 0.2rem;
        }
        .analysis-value {
            font-weight: 800;
            font-size: 1.02rem;
        }
        .interpretation-box {
            border-left: 4px solid rgba(20, 140, 90, 0.65);
            padding: 0.75rem 0.95rem;
            margin-top: 0.8rem;
            border-radius: 12px;
            background: rgba(20, 140, 90, 0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def make_card_options():
    cards = []

    for rank in RANKS:
        for suit_symbol, suit_code in SUITS.items():
            cards.append({
                "label": f"{rank}{suit_symbol}",
                "code": f"{rank}{suit_code}",
            })

    return cards


CARD_OPTIONS = make_card_options()
CARD_LABELS = [card["label"] for card in CARD_OPTIONS]
LABEL_TO_CODE = {
    card["label"]: card["code"]
    for card in CARD_OPTIONS
}


def labels_to_card_text(labels):
    return " ".join(LABEL_TO_CODE[label] for label in labels)


def render_card_pills(title, cards):
    st.markdown(f"**{title}**")

    if not cards:
        st.markdown('<span class="small-muted">No cards selected</span>', unsafe_allow_html=True)
        return

    html = "".join(f'<span class="card-pill">{card}</span>' for card in cards)
    st.markdown(html, unsafe_allow_html=True)


def render_q_values(q_values, legal_actions, recommended_action):
    st.subheader("Model Q Values")

    legal_values = [q_values[action] for action in legal_actions]

    if legal_values:
        min_q = min(legal_values)
        max_q = max(legal_values)
    else:
        min_q = 0.0
        max_q = 1.0

    q_range = max(max_q - min_q, 1e-6)

    for action in ["fold", "check", "call", "bet", "raise"]:
        is_legal = action in legal_actions
        is_recommended = action == recommended_action
        row_class = "action-row action-row-strong" if is_recommended else "action-row"

        legal_text = "Legal" if is_legal else "Illegal"
        recommend_text = "Recommended" if is_recommended else ""
        q_value = q_values[action]

        if is_legal:
            normalized = (q_value - min_q) / q_range
            progress_value = max(0.0, min(1.0, normalized))
        else:
            progress_value = 0.0

        st.markdown(
            f"""
            <div class="{row_class}">
                <div style="display:flex; justify-content:space-between; gap:1rem; align-items:center;">
                    <div>
                        <b>{ACTION_LABELS[action]}</b><br>
                        <span class="small-muted">{legal_text} {recommend_text}</span>
                    </div>
                    <div style="font-weight:800;">{q_value:.4f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(progress_value)


def render_hand_analysis(draw_info):
    st.subheader("AI Hand Analysis")

    target_hands = ", ".join(draw_info["target_hands"])
    draw_types = ", ".join(draw_info["draw_types"])

    st.markdown(
        f"""
        <div class="analysis-grid">
            <div class="analysis-item">
                <div class="analysis-label">Current Hand</div>
                <div class="analysis-value">{draw_info["current_hand"]}</div>
            </div>
            <div class="analysis-item">
                <div class="analysis-label">Target Hands</div>
                <div class="analysis-value">{target_hands}</div>
            </div>
            <div class="analysis-item">
                <div class="analysis-label">Draw Types</div>
                <div class="analysis-value">{draw_types}</div>
            </div>
            <div class="analysis-item">
                <div class="analysis-label">Effective Outs</div>
                <div class="analysis-value">{draw_info["effective_outs"]}</div>
            </div>
            <div class="analysis-item">
                <div class="analysis-label">Useful Cards</div>
                <div class="analysis-value">{draw_info["useful_cards"]}</div>
            </div>
            <div class="analysis-item">
                <div class="analysis-label">Draw Quality</div>
                <div class="analysis-value">Flush: {draw_info["flush_draw_quality"]} / Straight: {draw_info["straight_draw_type"]}</div>
            </div>
        </div>
        <div class="interpretation-box">
            <b>Interpretation:</b><br>
            {draw_info["interpretation"]}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Texas Hold'em AI",
        page_icon="🃏",
        layout="wide",
    )

    inject_css()

    st.markdown('<div class="main-title">🃏 Texas Hold\'em AI Decision Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">选择手牌、公共牌和当前局势，让本地训练好的 DQN 模型给出动作建议。</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Settings")

        model_path = st.text_input(
            "Model path",
            value="models/dqn_final.pt",
        )

        st.markdown("---")
        st.caption("Action space")
        st.write("Fold / Check / Call / Bet / Raise")

        st.markdown("---")
        st.info(
            "当前模型来自本地简化训练环境，只适合学习、研究和本地测试，不代表真实扑克最优策略。"
        )

    left, right = st.columns([1.05, 1])

    with left:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.subheader("Cards")

        hole_cards = st.multiselect(
            "Your hole cards，必须选 2 张",
            CARD_LABELS,
            max_selections=2,
            placeholder="Choose exactly 2 cards",
        )

        board_cards = st.multiselect(
            "Board cards，选择 0 / 3 / 4 / 5 张",
            CARD_LABELS,
            max_selections=5,
            placeholder="Choose community cards",
        )

        duplicated_cards = set(hole_cards) & set(board_cards)

        render_card_pills("Selected hole cards", hole_cards)
        render_card_pills("Selected board cards", board_cards)

        if duplicated_cards:
            st.error(f"重复选择了牌：{', '.join(duplicated_cards)}")

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.subheader("Game State")

        street = st.selectbox(
            "Street",
            ["preflop", "flop", "turn", "river"],
            help="preflop=0张公共牌，flop=3张，turn=4张，river=5张。",
        )

        expected_count = EXPECTED_BOARD_COUNT[street]
        actual_count = len(board_cards)

        if actual_count != expected_count:
            st.warning(
                f"当前 Street 是 {street}，应选择 {expected_count} 张公共牌；"
                f"现在选择了 {actual_count} 张。"
            )
        else:
            st.success(f"Street 与公共牌数量匹配：{street} / {actual_count} 张公共牌。")

        col_a, col_b = st.columns(2)

        with col_a:
            pot = st.number_input(
                "Pot",
                min_value=0,
                max_value=100000,
                value=15,
                step=5,
            )

            current_bet = st.number_input(
                "Current highest bet",
                min_value=0,
                max_value=100000,
                value=10,
                step=5,
            )

            my_current_bet = st.number_input(
                "My current bet",
                min_value=0,
                max_value=100000,
                value=5,
                step=5,
            )

        with col_b:
            opponent_current_bet = st.number_input(
                "Opponent current bet",
                min_value=0,
                max_value=100000,
                value=10,
                step=5,
            )

            my_chips = st.number_input(
                "My chips",
                min_value=0,
                max_value=100000,
                value=995,
                step=5,
            )

            opponent_chips = st.number_input(
                "Opponent chips",
                min_value=0,
                max_value=100000,
                value=990,
                step=5,
            )

        expected_current_bet = get_expected_current_bet(
            my_current_bet=my_current_bet,
            opponent_current_bet=opponent_current_bet,
        )

        if int(current_bet) != expected_current_bet:
            st.warning(
                f"当前下注状态不一致：Current highest bet 应该等于 "
                f"max(My current bet, Opponent current bet) = {expected_current_bet}，"
                f"但现在输入的是 {int(current_bet)}。"
            )
        else:
            st.success(
                f"下注状态匹配：Current highest bet = {expected_current_bet}。"
            )

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    ask_clicked = st.button("Ask AI", type="primary", use_container_width=True)

    if ask_clicked:
        if len(hole_cards) != 2:
            st.error("必须选择 exactly 2 张手牌。")
            return

        if len(board_cards) not in [0, 3, 4, 5]:
            st.error("公共牌数量必须是 0、3、4 或 5。")
            return

        expected_count = EXPECTED_BOARD_COUNT[street]
        actual_count = len(board_cards)

        if actual_count != expected_count:
            st.error(
                f"输入状态不一致：当前 Street 是 {street}，公共牌数量必须是 "
                f"{expected_count} 张；你现在选了 {actual_count} 张。"
            )
            return

        expected_current_bet = get_expected_current_bet(
            my_current_bet=my_current_bet,
            opponent_current_bet=opponent_current_bet,
        )

        if int(current_bet) != expected_current_bet:
            st.error(
                f"输入状态不一致：Current highest bet 必须等于 "
                f"max(My current bet, Opponent current bet) = {expected_current_bet}；"
                f"你现在输入的是 {int(current_bet)}。"
            )
            return

        if set(hole_cards) & set(board_cards):
            st.error("手牌和公共牌不能重复。")
            return

        hole_text = labels_to_card_text(hole_cards)
        board_text = labels_to_card_text(board_cards)

        try:
            result = predict_action(
                model_path=model_path,
                hole_cards_text=hole_text,
                board_cards_text=board_text,
                pot=int(pot),
                current_bet=int(current_bet),
                my_chips=int(my_chips),
                opponent_chips=int(opponent_chips),
                my_current_bet=int(my_current_bet),
                opponent_current_bet=int(opponent_current_bet),
                street=street,
            )

        except Exception as error:
            st.exception(error)
            return

        recommended_action = result["recommended_action"]
        legal_actions = result["legal_actions"]
        q_values = result["q_values"]

        observation = result["observation"]
        draw_info = analyze_draws(
            hole_cards=observation["hole_cards"],
            board_cards=observation["board"],
        )

        st.markdown(
            f"""
            <div class="result-box">
                <div class="small-muted">Recommended Action</div>
                <div class="recommended-action">{recommended_action.upper()}</div>
                <div class="small-muted">Legal actions: {', '.join(legal_actions)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        result_left, result_right = st.columns([1, 1])

        with result_left:
            render_q_values(q_values, legal_actions, recommended_action)

        with result_right:
            render_hand_analysis(draw_info)

        st.markdown("---")
        st.subheader("Input Summary")

        summary_rows = [
            ("Hole Cards", ", ".join(hole_cards)),
            ("Board", ", ".join(board_cards) if board_cards else "None"),
            ("Street", street),
            ("Pot", str(pot)),
            ("Current Bet", str(current_bet)),
            ("My Current Bet", str(my_current_bet)),
            ("Opponent Current Bet", str(opponent_current_bet)),
            ("My Chips", str(my_chips)),
            ("Opponent Chips", str(opponent_chips)),
        ]

        for label, value in summary_rows:
            st.markdown(f"**{label}:** {value}")

        st.caption(
            "Q value 越高表示模型在当前简化训练环境下越偏好该动作；非法动作不会被选择。"
        )


if __name__ == "__main__":
    main()