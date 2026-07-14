import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from networks.a2c_network import A2CNet
from config.settings import MAX_GRAD_NORM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class A2CAgent:
    def __init__(
        self,
        n_actions: int,
        discount: float,
        lr: float,
        value_loss_coef: float,
        entropy_coef: float,
        gae_lambda: float,
    ):
        self.n_actions = n_actions
        self.discount = discount
        self.lr = lr
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda

        self.net = A2CNet(n_actions).to(DEVICE)
        self.opt = optim.Adam(self.net.parameters(), lr=self.lr)

        self.stat_loss_sum = 0.0
        self.stat_val_sum = 0.0
        self.stat_steps = 0

    def get_action(self, state: np.ndarray) -> int:
        state = np.expand_dims(state, axis=0)
        s_t = torch.from_numpy(state).float().to(DEVICE)
        with torch.no_grad():
            logits, _ = self.net(s_t)
            probs = F.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            a = dist.sample()
        return int(a.item())

    def act(self, state: np.ndarray):
        state = np.expand_dims(state, axis=0)
        s_t = torch.from_numpy(state).float().to(DEVICE)
        logits, value = self.net(s_t)
        probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return int(action.item()), value, log_prob, entropy

    def update_rollout(self, rewards, dones, values, log_probs, entropies,
                       next_state, last_done: bool):
        T = len(rewards)
        if T == 0:
            return

        if last_done:
            last_value = torch.zeros(1, 1, device=DEVICE)
        else:
            ns = np.expand_dims(next_state, axis=0)
            ns_t = torch.from_numpy(ns).float().to(DEVICE)
            with torch.no_grad():
                _, last_value = self.net(ns_t)

        values_t = torch.cat(values + [last_value], dim=0).squeeze(-1)
        log_probs_t = torch.stack(log_probs).squeeze(-1)
        entropies_t = torch.stack(entropies).squeeze(-1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=DEVICE)

        advantages = torch.zeros(T, dtype=torch.float32, device=DEVICE)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones_t[t]
            delta = (
                rewards_t[t]
                + self.discount * values_t[t + 1].detach() * mask
                - values_t[t].detach()
            )
            gae = delta + self.discount * self.gae_lambda * mask * gae
            advantages[t] = gae

        returns = advantages + values_t[:-1].detach()

        policy_loss = -(log_probs_t * advantages.detach()).mean()
        value_loss = F.mse_loss(values_t[:-1], returns)
        entropy_loss = entropies_t.mean()

        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRAD_NORM)
        self.opt.step()

        self.stat_loss_sum += loss.item()
        self.stat_val_sum += values_t[:-1].detach().mean().item()
        self.stat_steps += 1

    def get_and_reset_stats(self):
        if self.stat_steps == 0:
            return 0.0, 0.0
        m_loss = self.stat_loss_sum / self.stat_steps
        m_val = self.stat_val_sum / self.stat_steps
        self.stat_loss_sum = 0.0
        self.stat_val_sum = 0.0
        self.stat_steps = 0
        return m_loss, m_val
