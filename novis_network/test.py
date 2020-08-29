from collections import deque

import matplotlib.pyplot as plt
import numpy as np

import torch
from agent import Agent
from unityagents import UnityEnvironment


class TestRunner:
    def __init__(self, env_path: str):
        self.env = UnityEnvironment(file_name=env_path)
        self.brain_name = self.env.brain_names[0]
        self.agent = Agent(state_size=37, action_size=4, seed=42)

    def run(self) -> None:
        # load the weights from file
        self.agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

        rewards = 0

        for i in range(10):
            # reset the environment
            env_info = self.env.reset(train_mode=False)[self.brain_name]
            state = env_info.vector_observations[0]  # get the current state
            for j in range(200):
                action = self.agent.act(state)

                env_info = self.env.step(action)[
                    self.brain_name]  # send the action to the environment
                state = env_info.vector_observations[0]  # get the next state
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]  # see if episode has finished
                rewards += reward
                if reward > 0:
                    print("rewards: ", rewards)
                if done:
                    break

        self.env.close()


if __name__ == '__main__':
    TestRunner("./Banana_Linux/Banana.x86_64").run()
