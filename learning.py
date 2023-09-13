import itertools
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('./path/to/log')


class Actor(nn.Module):
    def __init__(self, state_num, action_num, hidden_layer, action_max,
                 init_w=3e-3, init_b=1e-2):
        super().__init__()
        self.action_max = action_max
        self.action_num = action_num

        self.linear1 = nn.Linear(state_num, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, hidden_layer)
        #self.linear2.weight.data.uniform_(-init_w, init_w)
        #self.linear2.bias.data.uniform_(-init_b, init_b)
        self.mean = nn.Linear(hidden_layer, action_num)
        self.log_std = nn.Linear(hidden_layer, action_num)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # 限制方差的大小
        return mean, log_std

    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).to(device)
        mean, log_std = self.forward(state)
        if deterministic:
            action = torch.tanh(mean) * self.action_max
        else:
            N = torch.randn(mean.shape).to(device)
            std = torch.exp(log_std)
            action = mean + N * std
            action = torch.tanh(action) * self.action_max
        return action

    def compute(self, state):
        mean, log_std = self.forward(state)
        N = torch.randn(mean.shape).to(device)
        std = torch.exp(log_std)
        u = mean + N * std  # 这里输出动作是为了得到a_用于更新critic
        action = torch.tanh(u)
        # log_prob = torch.distributions.Normal(mean, std).log_prob(mean + N * std) - torch.log(
        #    1 - action ** 2 + 1e-6) - np.log(self.action_max)
        # log_prob = torch.sum(log_prob, dim=1)  # 等会看一下维度，再进行调试
        log_prob = torch.distributions.Normal(mean, std).log_prob(u).sum(axis=1, keepdim=True) - (
                2 * (np.log(2) - u - F.softplus(-2 * u))).sum(
            axis=1, keepdim=True) - np.log(self.action_max)
        action = action * self.action_max  # 改了action 和 std的计算
       
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_num, action_num, hidden_layer, init_w=3e-3, init_b=1e-2):
        super().__init__()

        self.linear1 = nn.Linear(state_num + action_num, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, hidden_layer)
        self.linear3 = nn.Linear(hidden_layer, 1)
        #self.linear3.weight.data.uniform_(-init_w, init_w)
        #self.linear3.bias.data.uniform_(-init_b, init_b)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0

    def push(self, s, a, r, s_, done):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        self.buffer[self.position] = (s, a, r, s_, done)
        self.position = (self.position + 1) % self.buffer_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, done = map(np.stack, zip(*batch))
        return s, a, r, s_, done

    def __len__(self):
        return len(self.buffer)


class Agent:
    def __init__(self, state_num, action_num, action_max, hidden_layer, gamma,
                 actor_lr, critic_lr, tau, alpha, batch_size, buffer_size, auto, H):
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=device)
        self.auto = auto
        self.H = H

        self.actor_net = Actor(state_num, action_num, hidden_layer, action_max).to(device)
        self.critic_net1 = Critic(state_num, action_num, hidden_layer).to(device)
        self.critic_net2 = Critic(state_num, action_num, hidden_layer).to(device)
        self.q_params = itertools.chain(self.critic_net1.parameters(), self.critic_net2.parameters())
        self.critic_net_optimizer = optim.Adam(self.q_params, lr=self.critic_lr)
        self.critic_target_net1 = Critic(state_num, action_num, hidden_layer).to(device)
        self.critic_target_net2 = Critic(state_num, action_num, hidden_layer).to(device)
        self.actor_net_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)
        self.critic_net1_optimizer = optim.Adam(self.critic_net1.parameters(), lr=self.critic_lr)
        self.critic_net2_optimizer = optim.Adam(self.critic_net2.parameters(), lr=self.critic_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.actor_lr)
        for p in self.critic_target_net1.parameters():
            p.requires_grad = False
        for p in self.critic_target_net2.parameters():
            p.requires_grad = False
    def update(self):

        s, a, r, s_, done = self.replay_buffer.sample(self.batch_size)
        s = torch.FloatTensor(s).to(device)
        a = torch.FloatTensor(a).to(device)
        r = torch.FloatTensor(r).reshape(-1, 1).to(device)
        s_ = torch.FloatTensor(s_).to(device)
        done = torch.FloatTensor(done).reshape(-1, 1).to(device)

        with torch.no_grad():
            a_, log_prob_ = self.actor_net.compute(s_)
            log_prob_ = torch.reshape(log_prob_, (-1, 1))
            q1_ = self.critic_target_net1(s_, a_)
            q2_ = self.critic_target_net2(s_, a_)
            q_min_ = torch.min(q1_, q2_)
            target_q = r + self.gamma * (1 - done) * (q_min_ - self.alpha * log_prob_)

        q1 = self.critic_net1(s, a)
        q2 = self.critic_net2(s, a)

        loss_q1 = F.mse_loss(q1, target_q)
        loss_q2 = F.mse_loss(q2, target_q)
        # 这里的detach很关键，因为q1和q2在前面critic的训练时被丢弃，而且此时训练actor不需要更新q
        #loss_q = loss_q1 + loss_q2

        #self.critic_net_optimizer.zero_grad()
        #loss_q.backward()
        #self.critic_net_optimizer.step()
        self.critic_net1_optimizer.zero_grad()
        loss_q1.backward()
        self.critic_net1_optimizer.step()

        self.critic_net2_optimizer.zero_grad()
        loss_q2.backward()
        self.critic_net2_optimizer.step()

        a, log_prob = self.actor_net.compute(s)
        log_prob = torch.reshape(log_prob, (-1, 1))
        q1 = self.critic_net1(s, a)
        q2 = self.critic_net2(s, a)
        q_min = torch.min(q1, q2)
        loss_actor = (self.alpha * log_prob - q_min).mean()

        self.actor_net_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_net_optimizer.step()

        if self.auto:
            loss_alpha = (-self.log_alpha * log_prob.detach() - self.log_alpha * self.H).mean()
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        for target_param, param in zip(self.critic_target_net1.parameters(), self.critic_net1.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        for target_param, param in zip(self.critic_target_net2.parameters(), self.critic_net2.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
