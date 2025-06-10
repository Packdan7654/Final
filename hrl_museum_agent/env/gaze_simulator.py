import numpy as np

def simulate_dwell_ratio(persona):
    if persona == "Inquisitive":
        return np.clip(np.random.normal(0.8, 0.05), 0, 1)
    elif persona == "Passive":
        return np.clip(np.random.normal(0.5, 0.1), 0, 1)
    elif persona == "Distracted":
        return np.clip(np.random.normal(0.3, 0.15), 0, 1)
    elif persona == "Knowledgeable":
        return np.clip(np.random.normal(0.7, 0.1), 0, 1)
    return 0.5
