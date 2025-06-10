from transformers import AutoTokenizer, AutoModel
import torch

def load_roberta():
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModel.from_pretrained("roberta-base")
    return tokenizer, model

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # CLS token

def flatten_state_dict(state_dict):
    import numpy as np
    return np.concatenate([
        state_dict["dialogue_embedding"],
        state_dict["exhibit_embedding"],
        state_dict["turn_metadata"]
    ])
