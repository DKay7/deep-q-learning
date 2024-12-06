from torch._dynamo.convert_frame import config
from agent import Agent
from model import CNNEncoder, FCEncoder
from trainer import Trainer
import gymnasium as gym
import argparse
import ale_py
import flappy_bird_gymnasium

def play(model_path=None):
    env = gym.make("FlappyBird-v0",  render_mode="human")
    agent = Agent(env, FCEncoder)

    if model_path:
        agent.load_model(model_path)

    agent.play()


def train(config):
    env = gym.make("FlappyBird-v0")
    agent = Agent(env, FCEncoder)
    trainer = Trainer(agent, config)
    trainer.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action=argparse.BooleanOptionalAction)
    parser.add_argument("--config")
    parser.add_argument("--model")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.train:
        train(args.config)
    else:
        play(args.model)


if __name__ == "__main__":
    main()
