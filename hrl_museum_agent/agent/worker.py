import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.networks import PolicyNet

class WorkerPolicy:
    def __init__(self, input_dim=871, output_dim=10):
        self.policy = PolicyNet(input_dim, output_dim)

    def select(self, state_tensor):
        logits = self.policy(state_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()
