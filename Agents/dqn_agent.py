import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks.dueling_dqn import DuelQNet
from networks.noisy_dueling_dqn import NoisyDuelQNet
from config.settings import MAX_GRAD_NORM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent:
    def __init__(
        self,
        n_actions: int,
        memory_size: int,
        batch_size: int,
        discount: float,
        lr: float,
        model_path: str,
        noisy: bool = False,
        load_model: bool = False,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9996,
        epsilon_min: float = 0.1,
        target_update_interval: int = 1000,
    ):
        self.n_actions = n_actions
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.discount = discount
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.model_path = model_path
        self.target_update_interval = target_update_interval

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        if noisy:
            self.q_net = NoisyDuelQNet(n_actions).to(DEVICE)
            self.target_net = NoisyDuelQNet(n_actions).to(DEVICE)
        else:
            self.q_net = DuelQNet(n_actions).to(DEVICE)
            self.target_net = DuelQNet(n_actions).to(DEVICE)

        if load_model and os.path.isfile(self.model_path):
            print(f"[DQN] Loading model from {self.model_path}")
            state_dict = torch.load(self.model_path, map_location=DEVICE)
            self.q_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)
            self.epsilon = self.epsilon_min
        else:
            print("[DQN] Initializing new model")
            self.update_target_net()

        self.opt = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.train_steps = 0

        self.stat_loss_sum = 0.0
        self.stat_q_sum = 0.0
        self.stat_steps = 0

    def get_action(self, state: np.ndarray) -> int:
        if self.epsilon > 0.0 and np.random.rand() < self.epsilon:
            return random.randrange(self.n_actions)
        state = np.expand_dims(state, axis=0)
        s_t = torch.from_numpy(state).float().to(DEVICE)
        with torch.no_grad():
            q_vals = self.q_net(s_t)
        return int(torch.argmax(q_vals, dim=1).item())

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def append_memory(self, s0, a0, r_n, ns, done, n_steps):
        self.memory.append((s0, a0, r_n, ns, done, n_steps))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        batch = np.array(batch, dtype=object)

        states = np.stack(batch[:, 0]).astype(np.float32)
        actions = batch[:, 1].astype(np.int64)
        rewards_n = batch[:, 2].astype(np.float32)
        next_states = np.stack(batch[:, 3]).astype(np.float32)
        dones = batch[:, 4].astype(bool)
        n_steps_arr = batch[:, 5].astype(np.int64)

        idx = np.arange(self.batch_size)
        not_dones = ~dones

        with torch.no_grad():
            ns_t = torch.from_numpy(next_states).float().to(DEVICE)
            q_next_online = self.q_net(ns_t)
            best_actions = torch.argmax(q_next_online, dim=1).cpu().numpy()
            q_next_target = self.target_net(ns_t).cpu().numpy()
            next_state_values = q_next_target[idx, best_actions]

        q_targets = rewards_n.copy()
        if np.any(not_dones):
            gamma_ns = (self.discount ** n_steps_arr[not_dones])
            q_targets[not_dones] += gamma_ns * next_state_values[not_dones]

        q_targets_t = torch.from_numpy(q_targets).float().to(DEVICE)
        states_t = torch.from_numpy(states).float().to(DEVICE)
        q_all = self.q_net(states_t)
        action_values = q_all[idx, actions].float()

        self.opt.zero_grad()
        loss = self.criterion(action_values, q_targets_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), MAX_GRAD_NORM)
        self.opt.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        self.stat_loss_sum += loss.item()
        self.stat_q_sum += action_values.mean().item()
        self.stat_steps += 1

        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            self.update_target_net()

    def get_and_reset_stats(self):
        if self.stat_steps == 0:
            return 0.0, 0.0
        mean_loss = self.stat_loss_sum / self.stat_steps
        mean_q = self.stat_q_sum / self.stat_steps
        self.stat_loss_sum = 0.0
        self.stat_q_sum = 0.0
        self.stat_steps = 0
        return mean_loss, mean_q
