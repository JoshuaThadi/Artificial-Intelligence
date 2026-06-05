# Soft Actor-Critic

# Check out this link for the complete model explanation: https://spinningup.openai.com/en/latest/algorithms/sac.html

# Importing the libraries

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Setting the hyperparameters

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_dim = 256
actor_lr = 3e-4
critic_lr = 3e-4
alpha_lr = 3e-4
gamma = 0.99
tau = 0.005
buffer_size = 1e6
batch_size = 128
alpha = 0.2  # Entropy coefficient

# Implementing the Replay Buffer

class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=int(capacity))
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward, dtype=np.float32), np.array(next_state), np.array(done, dtype=np.float32)
    
    def __len__(self):
        return len(self.buffer)

# Building the Actor Network

class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        return action

# Building the Critic Network

class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Building the SAC Agent

class SACAgent:

    def __init__(self):
        self.actor = Actor()
        self.critic_1 = Critic()
        self.critic_2 = Critic()
        self.target_critic_1 = Critic()
        self.target_critic_2 = Critic()
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor.sample(state)
        return action.detach().numpy()[0]

    def update(self, batch_size, gamma=gamma, tau=tau, alpha=alpha):
        if len(self.replay_buffer) < batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1)
        with torch.no_grad():
            next_state_action = self.actor.sample(next_state)
            target_q1_next = self.target_critic_1(next_state, next_state_action)
            target_q2_next = self.target_critic_2(next_state, next_state_action)
            target_q_min = torch.min(target_q1_next, target_q2_next) - alpha * torch.log(1 - next_state_action.pow(2) + 1e-6)
            target_q = reward + (1 - done) * gamma * target_q_min
        # Update of the Critic 1 network
        current_q1 = self.critic_1(state, action)
        critic_1_loss = F.mse_loss(current_q1, target_q)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        # Update of the Critic 2 network
        current_q2 = self.critic_2(state, action)
        critic_2_loss = F.mse_loss(current_q2, target_q)
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        # Update of the Actor network
        entropy = torch.log(1 - self.actor.sample(state).pow(2) + 1e-6)
        actor_loss = (-self.critic_1(state, self.actor.sample(state)) + alpha * entropy).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Soft update of the Critic Target networks
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# Setting up the environment

env = gym.make("CarRacing-v2")

# Creating the agent

agent = SACAgent()

# Implementing the Training Loop

num_episodes = 100
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        agent.update(batch_size)
        state = next_state
        episode_reward += reward
    print(f"Episode {episode}: Total Reward: {episode_reward}")
