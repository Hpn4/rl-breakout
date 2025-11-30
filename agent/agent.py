import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import *
from agent.model import *

class BaseAgent():

    def __init__(self, network, memory, n_actions, device):
        cfg = load_config()
        self.cfg = cfg
        
        self.policy_net = network(cfg["env"]["num_stack"], n_actions).to(device)
        self.target_net = network(cfg["env"]["num_stack"], n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = memory
        self.device = device

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg["train"]["lr"])
        self.criterion = nn.SmoothL1Loss()

        # Set parameters
        self.batch_size = cfg["train"]["batch_size"]
        self.gamma = cfg["train"]["gamma"]
        self.replay_start = cfg["train"]["replay_start"]

        self.EPS_START = cfg["train"]["eps_start"]
        self.EPS_END = cfg["train"]["eps_end"]
        self.EPS_DECAY_FRAMES = cfg["train"]["eps_decay_frames"]

    # -------------------------
    # EPSILON-GREEDY
    # -------------------------
    def action(self, i_step, state):
        bias = i_step * (self.EPS_START - self.EPS_END) / self.EPS_DECAY_FRAMES
        self.eps = max(self.EPS_END, self.EPS_START - bias)

        if random.random() > self.eps:
            with torch.no_grad():
                return self.predict(state)
        else:
            action = torch.tensor(random.randint(0, 3)).unsqueeze(0)

        return action.to("cpu")

    # -------------------------
    # ABSTRACT
    # -------------------------
    def calculate_loss(self, states, next_states, actions, rewards, dones):
        # Return the loss
        pass

    # -------------------------
    # OPTIMIZE
    # -------------------------
    def optimize(self):
        if len(self.memory) < self.replay_start:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.stack(actions).view(-1,1).to(self.device)
        rewards = torch.stack(rewards).view(-1,1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).view(-1,1)

        loss = self.calculate_loss(states, next_states, actions, rewards, dones)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg["train"]["clip_gradient"])
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def evaluate_q(self, states):
        with torch.no_grad():
            return self.policy_net(states).max(1)[0].mean().item()

    def predict(self, state):
        with torch.no_grad():
            return self.policy_net(state.unsqueeze(0).to(self.device)).max(1)[1].to("cpu")


class DQNAgent(BaseAgent):

    def __init__(self, memory, n_actions, device, network=DQN):
        super().__init__(network, memory, n_actions, device)

    def get_next_values(self, next_states):
        return self.target_net(next_states).max(1)[0].view(-1,1)

    def calculate_loss(self, states, next_states, actions, rewards, dones):
        # Q(s,a)
        state_action_values = self.policy_net(states).gather(1, actions)

        # Target Q
        with torch.no_grad():
            next_state_values = self.get_next_values(next_states)
            td_target = rewards + self.gamma * next_state_values * (1 - dones)

        loss = self.criterion(state_action_values, td_target)

        return loss

class DoubleDQNAgent(DQNAgent):

    def __init__(self, memory, n_actions, device, network=DQN):
        super().__init__(memory, n_actions, device, network)

    def get_next_values(self, next_states):
        next_policy_actions = self.policy_net(next_states).argmax(1).view(-1, 1)

        return self.target_net(next_states).gather(1, next_policy_actions)

class C51Agent(BaseAgent):

    def __init__(self, memory, n_actions, device):
        cfg = load_config()

        self.n_atoms = cfg["model"]["n_atoms"]
        self.Vmin = cfg["model"]["v_min"]
        self.Vmax = cfg["model"]["v_max"]
        self.delta_z = (self.Vmax - self.Vmin) / (self.n_atoms - 1)
        self.z = torch.linspace(self.Vmin, self.Vmax, self.n_atoms).to(device)

        super().__init__(C51DQN, memory, n_actions, device)

    def evaluate_q(self, states):
        with torch.no_grad():
            dist = self.policy_net(states)  # [batch, n_actions, n_atoms]
            q_values = (dist * self.z).sum(dim=2)  # [batch, n_actions]
            return q_values.max(1)[0].mean().item()
            # return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

    def predict(self, state):
        with torch.no_grad():
            dist = self.policy_net(state.unsqueeze(0).to(self.device))
            q_values = (dist * self.z).sum(dim=2)
            return q_values.argmax(dim=1).to("cpu")
            # return (self.policy_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

    def calculate_loss2(self, states, next_states, actions, rewards, dones):
        batch_size = states.size(0)

        with torch.no_grad():
            next_dist = self.target_net(next_states)
            next_q = (next_dist * self.z).sum(dim=2)
            next_actions = next_q.argmax(dim=1)
            next_dist = next_dist[range(batch_size), next_actions]

            Tz = rewards + self.gamma * self.z * (1 - dones)
            Tz = Tz.clamp(self.Vmin, self.Vmax)
            b = (Tz - self.Vmin) / self.delta_z
            
            l = b.floor().long().clamp(0, self.n_atoms - 1)
            u = b.ceil().long().clamp(0, self.n_atoms - 1)

            offset = torch.linspace(0, (batch_size - 1) * self.n_atoms, batch_size, 
                        device=self.device).long().unsqueeze(1)

            l_flat = (l + offset).view(-1)
            u_flat = (u + offset).view(-1)

            m = torch.zeros(batch_size * self.n_atoms, device=self.device)

            m.index_add_(0, l_flat, (next_dist * (u.float() - b)).view(-1))
            m.index_add_(0, u_flat, (next_dist * (b - l.float())).view(-1))

            eq = (l == u)
            m.index_add_(0, (l_flat)[eq.view(-1)], (next_dist.view(-1)[eq.view(-1)]))

            m = m.view(batch_size, self.n_atoms)
            m = m / (m.sum(1, keepdim=True) + 1e-8)

        dist_pred = self.policy_net(states)
        dist_pred = dist_pred.gather(1, actions.unsqueeze(-1).expand(-1, -1, self.n_atoms)).squeeze(1)

        loss = -(m * dist_pred.log()).sum(1).mean()

        return loss

    def calculate_loss(self, states, next_states, actions, rewards, dones):
        batch_size = states.size(0)

        log_ps = self.policy_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
        log_ps_a = log_ps[range(batch_size), actions]  # log p(s_t, a_t; θonline)

        with torch.no_grad():
            # Calculate nth next state probabilities
            pns = self.policy_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
            dns = self.z.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]

            pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
            pns_a = pns[range(batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

            # Compute Tz (Bellman operator T applied to z)
            Tz = rewards + self.gamma * self.z * (1 - dones)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.n_atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(batch_size, self.n_atoms)
            offset = torch.linspace(0, ((batch_size - 1) * self.n_atoms), batch_size).unsqueeze(1).expand(batch_size, self.n_atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        loss = -torch.sum(m * log_ps_a, 1).mean()  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))

        return loss

def build_agent(memory, n_actions, device):
    cfg = load_config()

    if cfg["model"]["C51"]:
        return C51Agent(memory, n_actions, device)

    if cfg["model"]["dueling"]:
        network = DuellingDQN
    else:
        network = DQN

    if cfg["model"]["double"]:
        agent = DoubleDQNAgent
    else:
        agent = DQNAgent

    return agent(memory, n_actions, device, network=network)
