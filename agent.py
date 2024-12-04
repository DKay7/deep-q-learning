from typing import Tuple, Type
import gymnasium as gym
import random
import torch 
from model import BaseEncoderType, DeepQNet


class Agent:
    def __init__(self, env: gym.Env, encoder_type: Type[BaseEncoderType], seed=42):
        self.seed = seed
        self.env = env
        print(env.observation_space.shape)
        encoder = encoder_type(env.observation_space.shape)
        self.model = DeepQNet(encoder, env.action_space.n)
        self.device = self.model.get_device()

    def get_random_action(self) -> torch.Tensor:
        return torch.tensor(self.env.action_space.sample(), dtype=torch.int64, device=self.device)

    def get_model_action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # adding batch dimension
            action = self.model(state.unsqueeze(0)).squeeze().argmax()
        
        return action

    def get_action_e_greedy(self, state, epsilon) -> torch.Tensor:
        if random.random() < epsilon:
            return self.get_random_action()
        else:
            return self.get_model_action(state)
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        new_state, reward, terminated, _, _ = self.env.step(action.item())
        new_state = torch.tensor(new_state, dtype=torch.float, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        terminated = torch.tensor(terminated, dtype=torch.float, device=self.device)
        
        return (new_state, reward, terminated)

    def reset_env(self) -> torch.Tensor:
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float, device=self.device)

        return state

    def play(self):
        self.model.eval()
        state = self.reset_env()

        while True:
            action = self.get_model_action(state) 
            
            state, _, terminated = self.step(action)
            
            if terminated:
                break

        self.env.close()

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

