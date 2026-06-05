# Proximal Policy Optimization (PPO)

# Check out this link for the complete model explanation: https://spinningup.openai.com/en/latest/algorithms/ppo.html

# Importing the libraries

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

# Setting the hyperparameters

hidden_dim = 256
lr_actor = 3e-4
lr_critic = 1e-3
gamma = 0.99
gae_lambda = 0.95
ppo_epochs = 10
mini_batch_size = 64
ppo_clip = 0.2
buffer_size = 2048
update_timestep = buffer_size
action_std = 0.5  # Standard deviation for action exploration

# Building the Actor-Critic Network

class ActorCritic(nn.Module):

    def __init__(self, num_actions):
        super(ActorCritic, self).__init__()
        # Common layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
        # Actor layers
        self.fc_actor = nn.Linear(hidden_dim, num_actions)
        self.log_std = nn.Parameter(torch.zeros(num_actions))
        # Critic layers
        self.fc_critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # Actor
        action_mean = self.fc_actor(x)
        action_var = torch.exp(self.log_std.expand_as(action_mean))
        cov_mat = torch.diag_embed(action_var)
        # Critic
        value = self.fc_critic(x)
        return action_mean, cov_mat, value

# Implementing the Memory Buffer

class Memory:

    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

# Building the PPO Agent

class PPO:

    def __init__(self, num_actions):
        self.policy = ActorCritic(num_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.policy_old = ActorCritic(num_actions)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            action_mean, action_var, _ = self.policy_old(state)
            dist = MultivariateNormal(action_mean, action_var)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        return action.detach().numpy(), action_logprob.detach()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # Converting list to tensor
        old_states = torch.stack(memory.states).detach()
        old_actions = torch.stack(memory.actions).detach()
        old_logprobs = torch.stack(memory.logprobs).detach()
        # Optimizing policy for K epochs
        for _ in range(ppo_epochs):
            # Evaluating old actions and values
            action_means, action_vars, state_values = self.policy(old_states)
            dists = MultivariateNormal(action_means, action_vars)
            logprobs = dists.log_prob(old_actions)
            dist_entropy = -logprobs.mean()
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-ppo_clip, 1+ppo_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            # Taking gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        # Copying new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

# Preprocessing the states

def preprocess_state(state):
    state = np.ascontiguousarray(state, dtype=np.float32) / 255
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

# Setting up the environment

env = gym.make('CarRacing-v2')

# Creating the memory

memory = Memory()

# Creating the agent

ppo = PPO(env.action_space.shape[0])

# Implementing the Training Loop

state = env.reset()
state = preprocess_state(state)
for t in range(1, update_timestep+1):
    action, action_logprob = ppo.select_action(state)
    next_state, reward, done, _ = env.step(action)
    next_state = preprocess_state(next_state)
    memory.states.append(state)
    memory.actions.append(torch.tensor(action))
    memory.logprobs.append(action_logprob)
    memory.rewards.append(reward)
    memory.is_terminals.append(done)
    state = next_state
    if done:
        state = env.reset()
        state = preprocess_state(state)
    if t % update_timestep == 0:
        ppo.update(memory)
        memory.clear_memory()
