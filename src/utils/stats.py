import numpy as np
import json, os
from typing import List, Dict
from scipy import stats

def set_seed(seed: int = 42):
    import torch, random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Set to {seed}")

def save_results(results: Dict, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[Stats] Saved → {filepath}")