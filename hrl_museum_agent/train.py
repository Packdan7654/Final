from agent.manager import ManagerPolicy
from agent.worker import WorkerPolicy
from env.persona_env import MuseumDialogueEnv
import torch
import numpy as np

def flatten_state(obs):
    import numpy as np
    return torch.tensor(
        np.concatenate([
            obs["dialogue_embedding"],
            obs["exhibit_embedding"],
            [obs["dwell_ratio"]],
            obs["turn_metadata"]
        ]),
        dtype=torch.float32
    ).unsqueeze(0)  # Make it batch-like


def train_agent():
    env = MuseumDialogueEnv()
    manager = ManagerPolicy()
    worker = WorkerPolicy()

    for episode in range(100):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            state_tensor = flatten_state(obs)

            
            print("Shape of state_tensor:", state_tensor.shape)
            print("Raw components:")
            print(" - dialogue:", obs["dialogue_embedding"].shape)
            print(" - exhibit:", obs["exhibit_embedding"].shape)
            print(" - dwell:", np.array([obs["dwell_ratio"]]).shape)
            print(" - metadata:", obs["turn_metadata"].shape)

            option = manager.select(state_tensor)
            action = worker.select(state_tensor)

            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

        print(f"Episode {episode}: {total_reward:.2f}")



