import os
import logging
import torch
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt


class StatsWriter:
    def __init__(self, config_name, config_file_dir, 
                 log_level=logging.INFO,
                 graph_update_time_sec=300,
                 log_filename="run.log", 
                 model_filename="model.pt", 
                 graph_filename="stats.png"):

        self.config_name = config_name


        self.run_path = os.path.join(config_file_dir, config_name)
        os.makedirs(self.run_path, exist_ok=True)

        self.log_filename = os.path.join(self.run_path, log_filename)
        self.model_filename = os.path.join(self.run_path, model_filename)
        self.graph_filename = os.path.join(self.run_path, graph_filename)

        logging.basicConfig(filename=self.log_filename, level=log_level)
        self.logger = logging.getLogger(__name__)

        self.best_reward = 0
        self.graph_update_time_sec = graph_update_time_sec
        self.last_time_graph_update = datetime.now()

    def save_model_if_needed(self, model, episode_reward):
        if episode_reward <= self.best_reward:
            return
        
        self.best_reward = episode_reward
        self.logger.info(f"Saving model with new best reward = {self.best_reward}")
        self.save_model(model)

    def plot_graph_if_needed(self, stats):
        if datetime.now() - self.last_time_graph_update > timedelta(seconds=self.graph_update_time_sec):
            self.plot_graph(stats)

    def save_model(self, model: torch.nn.Module):
        torch.save(model.state_dict(), self.model_filename)
    
    def plot_graph(self, stats):
        epoches = []
        epsilons = []
        rewards = []

        for epoch, stats_dict in stats:
            epoches.append(epoch)
            epsilons.append(stats_dict['epsilon'])
            rewards.append(stats_dict['reward'].item())
        
        mean_rewards = np.convolve(np.array(rewards), np.ones(50) / 50, mode='valid')

        fig = plt.figure(1)
        
        plt.subplot(121) 
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        plt.subplot(122) 
        plt.ylabel('Epsilon Decay')
        plt.plot(epoches, epsilons)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.graph_filename)
        plt.close(fig)

