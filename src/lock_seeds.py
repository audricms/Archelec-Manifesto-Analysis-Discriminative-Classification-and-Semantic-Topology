import random
import os
import numpy as np
import torch
from transformers import set_seed


def lock_seeds(seed=42):
    """
    Locks all random number generators across Python, NumPy, Scikit-Learn,
    PyTorch, and Hugging Face to ensure complete reproducibility.
    """
    print(f"Locking all stochastic processes to seed: {seed}")

    # 1. Standard Python library
    random.seed(seed)

    # 2. Python environment (affects hashing of dictionaries/strings)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 3. NumPy (Used heavily by Scikit-learn and UMAP)
    np.random.seed(seed)

    # 4. PyTorch (CPU & Apple Silicon / CUDA)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # 5. Hugging Face Transformers
    set_seed(seed)

    # 6. Force PyTorch to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
