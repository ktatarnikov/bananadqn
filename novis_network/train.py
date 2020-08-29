from collections import deque
from typing import Any, Callable, Dict, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

import torch
from agent import Agent
from unityagents import UnityEnvironment


class TrainRunner:
    def __init__(self, env_path: str):
        self.env = UnityEnvironment(file_name=env_path)
        self.brain_name = self.env.brain_names[0]
        self.agent = Agent(state_size=37, action_size=4, seed=42)

    def run(self,
            n_episodes: int = 1800,
            max_t: int = 1000,
            eps_start: float = 1.0,
            eps_end: float = 0.01,
            eps_decay: float = 0.995) -> Sequence[float]:
        """Deep Q-Learning.

        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start  # initialize epsilon
        for i_episode in range(1, n_episodes + 1):
            env_info = self.env.reset(
                train_mode=True)[self.brain_name]  # reset the environment
            state = env_info.vector_observations[0]  # get the current state
            score = 0
            for t in range(max_t):
                action = self.agent.act(state, eps)

                env_info = self.env.step(action)[
                    self.brain_name]  # send the action to the environment
                next_state = env_info.vector_observations[
                    0]  # get the next state
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]  # see if episode has finished

                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            eps = max(eps_end, eps_decay * eps)  # decrease epsilon

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)),
                  end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                    i_episode, np.mean(scores_window)))
            if np.mean(scores_window) >= 13.0:
                print(
                    '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'
                    .format(i_episode - 100, np.mean(scores_window)))
                torch.save(self.agent.qnetwork_local.state_dict(),
                           'checkpoint.pth')
                break
        return scores

    def close(self) -> None:
        self.env.close()


def plot_scores(scores: Sequence[float]) -> None:
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == '__main__':
    trainer = TrainRunner("./Banana_Linux/Banana.x86_64")
    scores = trainer.run()
    trainer.close()
    plot_scores(scores)
