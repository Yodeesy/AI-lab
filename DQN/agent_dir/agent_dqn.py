import os
import random
import copy
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from agent_dir.agent import Agent
from collections import deque
from tqdm import tqdm
import logging


class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        '''
        初始化Q网network
        '''
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        '''
        forward函数
        '''
        x = F.relu(self.fc1(inputs))
        x = self.fc2(x)
        return x

    # 更换计算设备
    def set_device(self, device: torch.device) -> torch.nn.Module:
        _model = self.to(device)
        _model.device = device
        return _model

    # 保存模型
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    # 加载模型
    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class ReplayBuffer:
    def __init__(self, buffer_size):
        '''
        初始化ReplayBuffer
        '''
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size

    def __len__(self):
        return len(self.buffer)

    def full(self):
        return len(self.buffer) == self.buffer_size

    def push(self, *transition):
        '''
        向buffer中添加一条数据
        '''
        self.buffer.append(transition)

    def sample(self, batch_size):
        '''
        根据参数batch_size从buffer中随机采样batch_size条数据
        '''
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def clean(self):
        self.buffer.clear()


class AgentDQN(Agent):
    def __init__(self, env, args):
        super(AgentDQN, self).__init__(env)
        self.args = args
        self.env_name = args.env_name  # 获取环境名称
        self.device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
        self.input_size = env.observation_space.shape[0]
        self.output_size = env.action_space.n
        self.hidden_size = args.hidden_size

        self.behavior_network = QNetwork(self.input_size, self.hidden_size, self.output_size).set_device(self.device)
        self.target_network = QNetwork(self.input_size, self.hidden_size, self.output_size).set_device(self.device)
        self.target_network.load_state_dict(self.behavior_network.state_dict())

        self.optimizer = optim.Adam(self.behavior_network.parameters(), lr=args.lr)
        self.replay_buffer = ReplayBuffer(args.buffer_size)
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.update_step = args.update_step
        self.max_episode_len = args.max_episode_len
        self.n_iter = args.n_iter
        # 修改模型保存路径的初始化
        self.model_path_base = "./model"
        os.makedirs(self.model_path_base, exist_ok=True)
        self.model_path = os.path.join(self.model_path_base, f"best_{self.env_name}.pt")
        self.log_path = args.log_path
        self.scaler = GradScaler()
        self.use_amp = args.use_cuda and torch.cuda.is_available()

        # Early Stopping
        self.patience = args.patience

        # 线性ε衰减参数
        self.epsilon_start = args.epsilon_start  # e.g. 1.0
        self.epsilon_end = args.epsilon_end      # e.g. 0.01
        self.epsilon_decay = args.epsilon_decay  # e.g. 50000
        self.epsilon = self.epsilon_start
        self.total_steps = 0  # 全局步数计数器

        # 添加用于存储损失的列表
        self.loss_history = []

        model_dir = os.path.dirname(self.model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        log_dir = os.path.dirname(self.log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def train(self):
        total_steps = 0
        patience = self.patience
        best_reward = float('-inf')
        patience_counter = 0

        for iter in tqdm(range(self.n_iter)):
            state, _ = self.env.reset()
            episode_reward = 0

            for step in range(self.max_episode_len):
                total_steps += 1
                action = self.make_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                if self.replay_buffer.full() and total_steps % self.update_step == 0:
                    loss = self.update()
                    if loss is not None:
                        self.loss_history.append(loss.item())

                if done:
                    break

            print(f'Iteration {iter}: Episode Reward = {episode_reward}, Best Reward = {best_reward}')

            # 提前终止策略
            if episode_reward >= best_reward:
                best_reward = episode_reward
                patience_counter = 0
                # 使用包含环境名称的路径保存模型
                self.behavior_network.save(self.model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at iteration {iter}. No improvement for {patience} iterations.")
                    break

        return self.loss_history  # 返回损失历史

    def make_action(self, observation, test=False):
        """
        Return predicted action of your agent
        """
        if test or random.random() > self.epsilon:
            state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_values = self.behavior_network(state)
            action = q_values.argmax().item()
        else:
            action = self.env.action_space.sample()

        # 随步数衰减 epsilon
        self.total_steps += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       np.exp(-1. * self.total_steps / self.epsilon_decay)

        return action

    def update(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        self.optimizer.zero_grad()

        loss = None
        # 使用AMP包裹前向传播
        with autocast(enabled=self.use_amp):
            q_values = self.behavior_network(states).gather(1, actions)
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            loss = F.mse_loss(q_values, target_q_values)

        # AMP反向传播与梯度更新
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        self.soft_update(self.tau)
        return loss

    def soft_update(self, tau):
        '''
        软更新（可选），tqu = 1时target网络完全更新为behavior网络
        '''
        for target_param, param in zip(self.target_network.parameters(), self.behavior_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)