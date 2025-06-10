import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env.gaze_simulator import simulate_dwell_ratio
from env.exhibit_knowledge import get_facts_for_template

class MuseumDialogueEnv(gym.Env):
    def __init__(self, persona="Inquisitive"):
        self.persona = persona
        self.observation_space = spaces.Dict({
            "dialogue_embedding": spaces.Box(-1, 1, shape=(768,), dtype=np.float32),
            "exhibit_embedding": spaces.Box(-1, 1, shape=(100,), dtype=np.float32),
            "dwell_ratio": spaces.Box(0, 1, shape=(), dtype=np.float32),
            "turn_metadata": spaces.Box(0, 100, shape=(2,), dtype=np.float32)
        })
        self.action_space = spaces.Discrete(20)  # Template IDs

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.used_facts = set()
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "dialogue_embedding": np.random.randn(768).astype(np.float32),
            "exhibit_embedding": np.random.randn(100).astype(np.float32),
            "dwell_ratio": simulate_dwell_ratio(self.persona),
            "turn_metadata": np.array([self.current_step, 0.0], dtype=np.float32)
        }

    def step(self, action):
        facts = get_facts_for_template(action)
        new_facts = set(facts) - self.used_facts
        novelty_reward = 0.3 * len(new_facts) 
        self.used_facts.update(new_facts)

        dwell_ratio = simulate_dwell_ratio(self.persona)
        engagement_reward = dwell_ratio

        total_reward = engagement_reward + novelty_reward
        self.current_step += 1
        done = self.current_step >= 20

        return self._get_obs(), total_reward, done, False, {}
