import itertools
import torch
import yaml
from agent import Agent
from memory import ReplayMemory
from model import CNNEncoder, FCEncoder
import gymnasium as gym
from copy import deepcopy
from torch import nn
import os
from writer import StatsWriter
from datetime import datetime, timedelta


class Trainer:
    def __init__(self, agent: Agent, config_name: str, 
                 config_file_dir: str | os.PathLike = "data", writer = None):
        
        self.writer = writer if writer else StatsWriter(config_name, config_file_dir)

        self.config_name = config_name
        hyperparams = self.read_config(os.path.join(config_file_dir, "hyperparams.yaml"), config_name)

        self.epsilon_init = hyperparams['epsilon_init']
        self.replay_memory_size = hyperparams['replay_memory_size']
        self.epsilon_decay_factor = hyperparams['epsilon_decay_factor']
        self.epsilon_min = hyperparams['epsilon_min']
        self.batch_size = hyperparams['batch_size']
        self.seed = hyperparams['seed']
        self.sync_target_rate = hyperparams['sync_target_rate']
        self.gamma_factor = hyperparams['gamma_factor']
        learning_rate = hyperparams['learning_rate']

        self.agent = agent
        if os.path.exists(os.path.join(self.writer.model_filename)):
            self.writer.logger.info(f"Loading agent model from {self.writer.model_filename}")
            self.agent.load_model(self.writer.model_filename)

        self.replay_memory = ReplayMemory(self.replay_memory_size, self.seed)
        self.stats_per_episode = []
        self.epsilon = self.epsilon_init

        self.target_model = deepcopy(agent.model)
        self.device = self.target_model.get_device()

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.agent.model.parameters(), learning_rate)
        

    def read_config(self, config_file_path, config_name):
        with open(config_file_path, 'r') as config_file:
            all_config = yaml.safe_load(config_file)
        
        return all_config[config_name]

    def save_stats_model(self, episode, model, stats):
        stats['reward'] = stats['reward'].cpu().item()
        self.writer.logger.info(f"Episode: {episode}, Reward: {stats['reward']}, Epsilon: {stats['epsilon']}")
        self.stats_per_episode.append((episode, stats))

        self.writer.plot_graph_if_needed(self.stats_per_episode)
        self.writer.save_model_if_needed(model, stats['reward'])
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.agent.model.state_dict())
    
    def optimize_agent(self):
        if len(self.replay_memory) < self.batch_size:
            return
        
        batch = self.replay_memory.sample(self.batch_size)

        state, action, new_state, reward, terminated = zip(*batch)
        state = torch.stack(state)
        action = torch.stack(action) # TODO: may be error in CNN-mode here
        new_state = torch.stack(new_state)
        reward = torch.stack(reward)
        terminated = torch.stack(terminated).flatten()
            
        with torch.no_grad():
            target_q = reward + (1 - terminated) * self.gamma_factor * self.target_model(new_state).max(dim=1).values
        
        # gather is used to pick predicted cost of chosen action,
        # i.e. actions = [0, 1, 0]
        # model output = [[1, 2], [3, 4], [5, 6]]
        # then the result will be: [1, 4, 5]
        current_q = self.agent.model(state).gather(dim=1, index=action.unsqueeze(dim=1)).squeeze()
        
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

    def train(self):
        for episode in itertools.count():

            if episode % self.sync_target_rate == 0:
                self.update_target_model()

            episode_reward = self.train_one_episode()
            self.optimize_agent()

            self.epsilon = max(self.epsilon * self.epsilon_decay_factor, self.epsilon_min)
            self.save_stats_model(episode, self.agent.model, {"reward": episode_reward, "epsilon": self.epsilon})
            

    def train_one_episode(self):
        terminated = False
        total_reward = 0.
        state = self.agent.reset_env()
        start_tyme = datetime.now()
        current_time = datetime.now()

        while not terminated and total_reward < 9999 and (current_time - start_tyme) < timedelta(minutes=5):
            action = self.agent.get_action_e_greedy(state, self.epsilon)
            new_state, reward, terminated  = self.agent.step(action)

            self.replay_memory.append((state, action, new_state, reward, terminated))

            total_reward += reward
            state = new_state

            current_time = datetime.now()

        return total_reward

if __name__ == "__main__":
    # import ale_py
    # import shimmy
    # env = gym.make("ALE/Alien-v5", render_mode="human")
    env = gym.make("CartPole-v1", render_mode='human')
    agent = Agent(env, FCEncoder)
    trainer = Trainer(agent, 'cartpole-1')
    trainer.train()
