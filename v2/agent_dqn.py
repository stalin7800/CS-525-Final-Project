#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
import os
import sys
from tqdm import tqdm
from collections import deque

import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim
from torch import amp  # Updated import for AMP

import matplotlib.pyplot as plt

from agent import Agent
from dqn_model import DQN

import wandb

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import warnings
warnings.filterwarnings("ignore")
"""
You can import any package and define any extra function as you need.
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

class ReplayBuffer:
    def __init__(self, max_size, state_shape, device, prioritized=False, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.max_size = max_size
        self.device = device
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.actions = np.zeros((max_size,), dtype=np.int64)
        self.rewards = np.zeros((max_size,), dtype=np.float32)
        self.next_states = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.dones = np.zeros((max_size,), dtype=np.float32)

        if self.prioritized:
            self.priorities = np.zeros((max_size,), dtype=np.float32)
            self.max_priority = 1.0

    def add(self, transition):
        state, action, reward, next_state, done = transition
        idx = self.ptr

        self.states[idx] = state.astype(np.float32)  # Ensure state is float32
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state.astype(np.float32)  # Ensure next_state is float32
        self.dones[idx] = done

        if self.prioritized:
            self.priorities[idx] = self.max_priority ** self.alpha

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        if self.prioritized:
            total = self.priorities.sum()
            probabilities = self.priorities[:self.size] / total
            # print("Probabilities: ", probabilities[:100])

            indices = np.random.choice(self.size, batch_size, p=probabilities)
            # print("Indices: ", indices)
            weights = (self.size * probabilities[indices]) ** (-self.beta)
            weights /= weights.max()
            # print("Weights: ", weights[:10])

            self.beta = min(1.0, self.beta + self.beta_increment)
            # print("Beta: ", self.beta)

            weights = torch.from_numpy(weights).float().to(self.device)

        else:
            indices = np.random.choice(self.size, batch_size, replace=False)
            weights = None

        # In the sample method of ReplayBuffer
        states = torch.from_numpy(self.states[indices]).permute(0, 3, 1, 2).to(self.device)
        actions = torch.from_numpy(self.actions[indices]).to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device)
        next_states = torch.from_numpy(self.next_states[indices]).permute(0, 3, 1, 2).to(self.device)
        dones = torch.from_numpy(self.dones[indices]).to(self.device)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        td_errors = td_errors.detach().cpu().numpy()
        self.priorities[indices] = (np.abs(td_errors) + 1e-5) ** self.alpha
        self.max_priority = max(self.max_priority, self.priorities[indices].max())
        # print("Prios: ", self.priorities[:100])

class NStep():
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.buffer = deque(maxlen=self.n)

    def compute_return(self, done):
        if len(self.buffer) < self.n and not done:
            return None
        R = 0
        for i, (_, _, reward, _, _) in enumerate(self.buffer):
            R += reward * self.gamma ** i 
        s, a, _, ns, d = self.buffer[0] 
        self.buffer.popleft()
        return s, a, R, ns, d 

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        """
        super(Agent_DQN, self).__init__(env)

        wandb.init(project="CS525-MsPacman")
        self.config = wandb.config
        self.config.update(vars(args))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.loss_fn = torch.nn.HuberLoss()

        self.steps = 0

        # Main parameters
        self.episodes = args.episodes
        self.update_target_net_freq = args.update_target_net_freq
        self.greedy_steps = 0
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.learning_rate = args.learning_rate
        self.epsilon = args.epsilon
        self.epsilon_min = args.epsilon_min
        self.epsilon_max = args.epsilon
        self.epsilon_decay_steps = args.epsilon_decay_steps

        #double
        self.no_double = args.no_double

        # Configure data directories and logging
        self.data_dir = args.data_dir
        self.model_name = args.model_name
        self.save_freq = args.save_freq
        self.write_freq = args.write_freq
        self.print_freq = args.print_freq
        self.rewards = []
        self.losses = []

        # Initialize buffer
        self.max_buffer_size = args.max_buffer_size
        self.buffer_start = args.buffer_start
        self.no_prio_replay = args.no_prio_replay

        state_shape = env.observation_space.shape
        self.buffer = ReplayBuffer(
            self.max_buffer_size,
            state_shape,
            self.device,
            prioritized=not self.no_prio_replay,
            alpha=args.prioritized_alpha,
            beta=args.prioritized_beta,
            beta_increment=args.prioritized_beta_increment
        )

        #N-Step buffer
        self.no_nstep = args.no_nstep
        self.n = args.n_step
        self.n_step = NStep(self.n, self.gamma)

        # Initialize model and target net
        self.q_net = DQN(args).to(self.device)
        self.target_net = DQN(args).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=self.learning_rate)

        if args.test_dqn or args.train_dqn_again:
            print('Loading trained model')
            checkpoint = torch.load(self.data_dir + self.model_name, map_location=self.device)
            self.q_net.load_state_dict(checkpoint['model_state_dict'])
            self.target_net.load_state_dict(checkpoint['model_state_dict'])
            self.epsilon = self.epsilon_min

        # Enable CUDNN Benchmarking for optimized convolution operations
        torch.backends.cudnn.benchmark = True

    def init_game_setting(self):
        """
        Initialize game settings if necessary.
        """
        pass

    def make_action(self, observation, num_actions=5, test=True):
        """
        Return predicted action of your agent.
        """
        obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).permute(0, 3, 1, 2).to(self.device, non_blocking=True)  # Ensure input is float32

        with amp.autocast(device_type=self.device.type):
            q_vals = self.q_net(obs_tensor)

        if not test and random.random() < self.epsilon:
            action = random.randint(0, num_actions - 1)
            # print("action space: ", self.env.action_space.sample())
        else:
            action = torch.argmax(q_vals, dim=1).item()

        return action

    def push(self, state, action, reward, next_state, done):
        """Push new data to buffer."""
        transition = state, action, reward, next_state, done
        if not self.no_nstep:
            self.n_step.buffer.append(transition)
            processed_transition = self.n_step.compute_return(done) #return for oldest transition in buffer
            if processed_transition: #if n-step buffer is filled
                self.buffer.add(transition)
        else:
            self.buffer.add(transition)
    
    def fill_buffer(self):
        state = self.env.reset()
        pbar = tqdm(total=self.buffer_start, desc="Filling Buffer", unit="steps")
        while self.buffer.size < self.buffer_start:
            action = self.make_action(state, test=False)
            next_state, reward, done, _, _ = self.env.step(action)
            self.push(state, action, reward, next_state, done)
            state = next_state
            pbar.update(1)
            if done:
                state = self.env.reset()
        pbar.close()
        print('Buffer filled')

    def update(self):
        states, actions, rewards, next_states, dones, indices, weights = self.buffer.sample(self.batch_size)

        qs = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        if self.no_double:  
            target_qs = self.target_net(next_states).max(dim=1)[0]
        else:
            next_qs = self.q_net(next_states)
            best_actions = torch.argmax(next_qs, dim=1)
            target_qs = self.target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
        
        targets = rewards + (1 - dones) * self.gamma * target_qs

        td_errors = targets.detach() - qs

        if not self.no_prio_replay:
            self.buffer.update_priorities(indices, td_errors)

        if weights is not None: #prio replay
            loss = (weights * self.loss_fn(qs, targets.detach())).mean()
        else:
            loss = self.loss_fn(qs, targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        nn_utils.clip_grad_norm_(self.q_net.parameters(), 1)
        self.optimizer.step()

        return loss.item(), td_errors

    def train(self):
        """
        Implement your training algorithm here.
        """
        all_rewards = []
        avg_rewards = []
        losses = []
        avg_losses = []
        epsilons = [self.epsilon]

        self.fill_buffer()

        for episode in tqdm(range(self.episodes)):
            state = self.env.reset()
            done = False

            total_reward = 0
            episode_loss = 0
            steps = 0

            while not done:
                action = self.make_action(state, test=False)
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.push(state, action, reward, next_state, done)

                total_reward += reward
                self.steps += 1
                steps += 1

                loss, td_errors = self.update()
                if loss:
                    episode_loss += loss

                if self.steps > 10000 and self.epsilon > self.epsilon_min:
                    self.update_epsilon()

                if self.steps % self.update_target_net_freq == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

                state = next_state
            
            while len(self.n_step.buffer) > 0: #handling remaining transitions in n-step buffer after done
                transition = self.n_step.compute_return(done)
                if transition:
                    self.buffer.add(transition)

            all_rewards.append(total_reward)
            avg_rewards.append(np.mean(all_rewards[-30:]))
            losses.append(episode_loss)
            avg_losses.append(np.mean(losses[-30:]))
            epsilons.append(self.epsilon)

            # Logging and saving
            if episode and episode % self.write_freq == 0:
                self.makePlots(all_rewards, avg_rewards, losses, epsilons)
                logger.info(f"Episode {episode+1}: Loss = {episode_loss}")
                logger.info(f"Episode {episode+1}: Avg Loss: {avg_losses[-1]}")
                logger.info(f"Episode {episode+1}: Avg Rewards = {avg_rewards[-1]}")
                logger.info(f"Episode {episode+1}: Epsilon = {self.epsilon}")
                logger.info(f"Episode {episode+1}: Steps this episode = {steps}")
                logger.debug(f"Last Batch TD Errors: {td_errors.mean()}")

                torch.save({
                    'model_state_dict': self.q_net.state_dict(),
                }, self.data_dir + self.model_name)
        torch.save({
            'model_state_dict': self.q_net.state_dict(),
        }, self.data_dir + self.model_name)

        wandb.finish()

    def makePlots(self, all_rewards, avg_rewards, losses, epsilons):
        plt.plot(avg_rewards, color='red')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.title('Avg Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Avg Rewards')
        plt.savefig(self.data_dir + 'avg_rewards.png')
        wandb.log({"Avg Rewards Plot": wandb.Image(self.data_dir + 'avg_rewards.png')})
        plt.clf()

        # Plot rewards
        plt.plot(all_rewards)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.title('Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.savefig(self.data_dir + 'rewards.png')
        wandb.log({"Rewards Plot": wandb.Image(self.data_dir + 'rewards.png')})
        plt.clf()

        plt.plot(np.log(losses))
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.title('Loss')
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.savefig(self.data_dir + 'loss.png')
        wandb.log({"Loss Plot": wandb.Image(self.data_dir + 'loss.png')})
        plt.clf()

        plt.close()

        # Log raw data to W&B
        for i, (reward, avg_reward, loss, epsilon) in enumerate(zip(all_rewards, avg_rewards, losses, epsilons)):
            wandb.log({
                "Episode": i + 1,
                "Reward": reward,
                "Avg Reward": avg_reward,
                "Loss": loss,
                "Epsilon": epsilon
            })

        # Write to file
        np.savetxt(self.data_dir + '/rewards.csv', all_rewards, delimiter=',', fmt='%d')
        np.savetxt(self.data_dir + '/avg_rewards.csv', avg_rewards, delimiter=',')
        np.savetxt(self.data_dir + '/loss.csv', losses, delimiter=',')
        np.savetxt(self.data_dir + '/epsilon.csv', epsilons, delimiter=',')

    def update_epsilon(self):
        self.epsilon -= (self.epsilon_max - self.epsilon_min) / self.epsilon_decay_steps
        self.epsilon = max(self.epsilon, self.epsilon_min)
        return self.epsilon
