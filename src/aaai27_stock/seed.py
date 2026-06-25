from __future__ import annotations

import os
import random
import numpy as np
import torch



def seed_everything(seed: int = 2027, deterministic: bool = True, benchmark: bool = False) -> None:
    # Keep CPU runs fast and reproducible in constrained notebook/container environments.
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = benchmark  # allow benchmark even in deterministic mode for speed
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
