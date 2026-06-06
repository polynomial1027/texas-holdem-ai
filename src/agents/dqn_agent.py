import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.holdem.encoder import get_state_dim, get_action_dim


class QNetwork(nn.Module):
    """
    DQN 的 Q 网络。

    输入：
        state vector, shape = 122

    输出：
        每个动作的 Q value, shape = 5

    动作顺序：
        0 fold
        1 check
        2 call
        3 bet
        4 raise
    """

    def __init__(self, state_dim=None, action_dim=None, hidden_dim=128):
        super().__init__()

        if state_dim is None:
            state_dim = get_state_dim()

        if action_dim is None:
            action_dim = get_action_dim()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """
    经验回放池。

    存储：
        state, action, reward, next_state, done, next_legal_action_mask
    """

    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state,
        action_index,
        reward,
        next_state,
        done,
        next_legal_action_mask,
    ):
        self.buffer.append((
            state,
            action_index,
            reward,
            next_state,
            done,
            next_legal_action_mask,
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones, next_masks = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        next_masks = torch.tensor(np.array(next_masks), dtype=torch.float32)

        return states, actions, rewards, next_states, dones, next_masks

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    最小 DQN Agent。

    目前特点：
    - epsilon-greedy 探索
    - target network
    - legal action mask
    - replay buffer
    """

    ACTIONS = ["fold", "check", "call", "bet", "raise"]

    ACTION_TO_INDEX = {
        "fold": 0,
        "check": 1,
        "call": 2,
        "bet": 3,
        "raise": 4,
    }

    INDEX_TO_ACTION = {
        0: "fold",
        1: "check",
        2: "call",
        3: "bet",
        4: "raise",
    }

    def __init__(
        self,
        state_dim=None,
        action_dim=None,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_capacity=50_000,
        batch_size=64,
        target_update_interval=100,
        device=None,
    ):
        if state_dim is None:
            state_dim = get_state_dim()

        if action_dim is None:
            action_dim = get_action_dim()

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.learn_step = 0

        self.device = torch.device(device)

        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def legal_actions_to_mask(self, legal_actions):
        mask = np.zeros(self.action_dim, dtype=np.float32)

        for action in legal_actions:
            index = self.ACTION_TO_INDEX[action]
            mask[index] = 1.0

        return mask

    def choose_action(self, state, legal_actions):
        """
        根据当前 state 和 legal_actions 选择动作。

        返回：
            action_string, action_index
        """
        legal_indices = [self.ACTION_TO_INDEX[action] for action in legal_actions]

        if not legal_indices:
            raise ValueError("No legal actions available.")

        # epsilon-greedy 随机探索
        if random.random() < self.epsilon:
            action_index = random.choice(legal_indices)
            return self.INDEX_TO_ACTION[action_index], action_index

        # 利用 Q 网络
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze(0)

        # 非法动作 mask 掉，避免 AI 选不能执行的动作
        masked_q_values = q_values.clone()

        for i in range(self.action_dim):
            if i not in legal_indices:
                masked_q_values[i] = -1e9

        action_index = int(torch.argmax(masked_q_values).item())
        return self.INDEX_TO_ACTION[action_index], action_index

    def store_transition(
        self,
        state,
        action_index,
        reward,
        next_state,
        done,
        next_legal_actions,
    ):
        next_mask = self.legal_actions_to_mask(next_legal_actions)

        self.replay_buffer.push(
            state=state,
            action_index=action_index,
            reward=reward,
            next_state=next_state,
            done=done,
            next_legal_action_mask=next_mask,
        )

    def learn(self):
        """
        从 replay buffer 中采样并更新 Q 网络。
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            next_masks,
        ) = self.replay_buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        next_masks = next_masks.to(self.device)

        current_q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_network(next_states)

            # mask 非法动作
            masked_next_q_values = next_q_values.masked_fill(
                next_masks == 0,
                -1e9,
            )

            max_next_q_values = masked_next_q_values.max(dim=1, keepdim=True)[0]

            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1

        if self.learn_step % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, path):
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]