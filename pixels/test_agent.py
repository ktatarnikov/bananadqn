from collections import deque

import torch
from pixels.agent import Agent
from pixels.train import LastNFrameBuffer
from unityagents import UnityEnvironment


class TestRunner:
    def __init__(self, env_path: str, checkpoint_path: str):
        self.env = UnityEnvironment(file_name=env_path)
        self.brain_name = self.env.brain_names[0]
        self.agent = Agent(state_size=37, action_size=4, seed=42)
        self.checkpoint_path = checkpoint_path

    def run(self) -> None:
        # load the weights from file
        model_state = torch.load(self.checkpoint_path)
        self.agent.qnetwork_local.load_state_dict(model_state)

        rewards = 0
        frame_buffer = LastNFrameBuffer(self.state_size, n=5)

        for i in range(10):
            # reset the environment
            env_info = self.env.reset(train_mode=False)[self.brain_name]
            frame = env_info.vector_observations[0]  # get the current state
            frame_buffer.add(frame)
            state = frame_buffer.get_last()
            for j in range(200):
                action = self.agent.act(state)
                # send the action to the environment
                env_info = self.env.step(action)[self.brain_name]
                # get the next frame
                next_frame = env_info.visual_observations[0]
                frame_buffer.add(next_frame)
                state = frame_buffer.get_last()
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]  # see if episode has finished
                rewards += reward
                if reward > 0:
                    print("rewards: ", rewards)
                if done:
                    break

        self.env.close()


if __name__ == '__main__':
    TestRunner("./Banana_Linux/Banana.x86_64", "./pixels_checkpoint.pth").run()
