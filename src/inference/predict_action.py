import argparse

import numpy as np
import torch

from src.holdem.cards import Card
from src.holdem.encoder import encode_observation
from src.agents.dqn_agent import DQNAgent



ACTION_NAMES = ["fold", "check", "call", "bet", "raise"]


# Expected number of board cards for each street
EXPECTED_BOARD_COUNT = {
    "preflop": 0,
    "flop": 3,
    "turn": 4,
    "river": 5,
}


RANK_MAP = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}


SUIT_MAP = {
    "s": "♠",
    "h": "♥",
    "d": "♦",
    "c": "♣",
    "S": "♠",
    "H": "♥",
    "D": "♦",
    "C": "♣",
    "♠": "♠",
    "♥": "♥",
    "♦": "♦",
    "♣": "♣",
}


def parse_card(text: str) -> Card:
    """
    把字符串转成 Card。

    支持格式：
        As, Ah, Ad, Ac
        Ks, Qh, Jd, Tc
        10s
        A♠, K♥
    """
    text = text.strip()

    if len(text) < 2:
        raise ValueError(f"Invalid card format: {text}")

    suit_text = text[-1]
    rank_text = text[:-1].upper()

    if rank_text not in RANK_MAP:
        raise ValueError(f"Invalid rank: {rank_text}")

    if suit_text not in SUIT_MAP:
        raise ValueError(f"Invalid suit: {suit_text}")

    return Card(rank=RANK_MAP[rank_text], suit=SUIT_MAP[suit_text])


def parse_cards(text: str):
    """
    把空格分隔的牌转成 Card 列表。

    例子：
        'As Kh'
        '2c 7d Js'
    """
    text = text.strip()

    if not text:
        return []

    return [parse_card(part) for part in text.split()]


def get_legal_actions(my_current_bet: int, current_bet: int, my_chips: int):
    """
    根据当前下注情况生成合法动作。

    这里要和 game.py 里的 get_legal_actions 保持一致。
    """
    to_call = current_bet - my_current_bet

    if to_call < 0:
        raise ValueError(
            f"Invalid betting state: current_bet={current_bet} is smaller than "
            f"my_current_bet={my_current_bet}."
        )

    legal_actions = []

    if to_call > 0:
        legal_actions.append("fold")
        legal_actions.append("call")

        if my_chips > to_call:
            legal_actions.append("raise")
    else:
        legal_actions.append("check")

        if my_chips > 0:
            legal_actions.append("bet")

    return legal_actions


def build_observation(
    hole_cards,
    board_cards,
    pot,
    current_bet,
    my_chips,
    opponent_chips,
    my_current_bet,
    opponent_current_bet,
    street,
):
    legal_actions = get_legal_actions(
        my_current_bet=my_current_bet,
        current_bet=current_bet,
        my_chips=my_chips,
    )

    observation = {
        "street": street,
        "current_player": "Player 1",
        "hole_cards": hole_cards,
        "board": board_cards,
        "pot": pot,
        "current_bet": current_bet,
        "player_chips": [my_chips, opponent_chips],
        "player_current_bets": [my_current_bet, opponent_current_bet],
        "player_folded": [False, False],
        "legal_actions": legal_actions,
    }

    return observation


def predict_action(
    model_path,
    hole_cards_text,
    board_cards_text,
    pot,
    current_bet,
    my_chips,
    opponent_chips,
    my_current_bet,
    opponent_current_bet,
    street,
):
    hole_cards = parse_cards(hole_cards_text)
    board_cards = parse_cards(board_cards_text)

    if len(hole_cards) != 2:
        raise ValueError("You must provide exactly 2 hole cards.")

    if len(board_cards) not in [0, 3, 4, 5]:
        raise ValueError("Board must contain 0, 3, 4, or 5 cards.")

    if street not in EXPECTED_BOARD_COUNT:
        raise ValueError(f"Unknown street: {street}")

    expected_board_count = EXPECTED_BOARD_COUNT[street]

    if len(board_cards) != expected_board_count:
        raise ValueError(
            f"Street {street} requires exactly {expected_board_count} board cards, "
            f"but got {len(board_cards)}."
        )

    expected_current_bet = max(int(my_current_bet), int(opponent_current_bet))

    if int(current_bet) != expected_current_bet:
        raise ValueError(
            f"current_bet must equal max(my_current_bet, opponent_current_bet) "
            f"= {expected_current_bet}, but got {int(current_bet)}."
        )

    all_cards = hole_cards + board_cards

    if len(set(all_cards)) != len(all_cards):
        raise ValueError("Duplicate cards detected.")

    observation = build_observation(
        hole_cards=hole_cards,
        board_cards=board_cards,
        pot=pot,
        current_bet=current_bet,
        my_chips=my_chips,
        opponent_chips=opponent_chips,
        my_current_bet=my_current_bet,
        opponent_current_bet=opponent_current_bet,
        street=street,
    )

    state = encode_observation(observation)

    agent = DQNAgent()
    agent.load(model_path)
    agent.epsilon = 0.0

    legal_actions = observation["legal_actions"]

    action, action_index = agent.choose_action(
        state=state,
        legal_actions=legal_actions,
    )

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
        device=agent.device,
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = agent.q_network(state_tensor).squeeze(0).detach().cpu().numpy()

    q_value_dict = {
        action_name: float(q_values[i])
        for i, action_name in enumerate(ACTION_NAMES)
    }

    legal_q_value_dict = {
        action_name: q_value_dict[action_name]
        for action_name in legal_actions
    }

    return {
        "recommended_action": action,
        "legal_actions": legal_actions,
        "q_values": q_value_dict,
        "legal_q_values": legal_q_value_dict,
        "observation": observation,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Use trained DQN model to predict Texas Hold'em action."
    )

    parser.add_argument("--model", default="models/dqn_final.pt")
    parser.add_argument("--hole", required=True, help="Your hole cards, e.g. 'As Kh'")
    parser.add_argument("--board", default="", help="Board cards, e.g. '2c 7d Js'")
    parser.add_argument("--pot", type=int, default=15)
    parser.add_argument("--current-bet", type=int, default=10)
    parser.add_argument("--my-chips", type=int, default=990)
    parser.add_argument("--opponent-chips", type=int, default=990)
    parser.add_argument("--my-current-bet", type=int, default=5)
    parser.add_argument("--opponent-current-bet", type=int, default=10)
    parser.add_argument(
        "--street",
        choices=["preflop", "flop", "turn", "river"],
        default="preflop",
    )

    args = parser.parse_args()

    result = predict_action(
        model_path=args.model,
        hole_cards_text=args.hole,
        board_cards_text=args.board,
        pot=args.pot,
        current_bet=args.current_bet,
        my_chips=args.my_chips,
        opponent_chips=args.opponent_chips,
        my_current_bet=args.my_current_bet,
        opponent_current_bet=args.opponent_current_bet,
        street=args.street,
    )

    print("=" * 70)
    print("Texas Hold'em AI Decision")
    print("=" * 70)

    obs = result["observation"]

    print("Street          :", obs["street"])
    print("Hole cards      :", obs["hole_cards"])
    print("Board           :", obs["board"])
    print("Pot             :", obs["pot"])
    print("Current bet     :", obs["current_bet"])
    print("My chips        :", obs["player_chips"][0])
    print("Opponent chips  :", obs["player_chips"][1])
    print("My current bet  :", obs["player_current_bets"][0])
    print("Opp current bet :", obs["player_current_bets"][1])
    print("Legal actions   :", result["legal_actions"])

    print("-" * 70)
    print("Q values:")

    for action_name in ACTION_NAMES:
        marker = ""
        if action_name == result["recommended_action"]:
            marker = "  <-- recommended"

        legal_marker = ""
        if action_name not in result["legal_actions"]:
            legal_marker = " [illegal]"

        print(
            f"{action_name:>5s}: "
            f"{result['q_values'][action_name]:10.4f}"
            f"{legal_marker}"
            f"{marker}"
        )

    print("-" * 70)
    print("Recommended action:", result["recommended_action"])
    print("=" * 70)


if __name__ == "__main__":
    main()