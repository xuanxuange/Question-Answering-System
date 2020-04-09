from pathlib import Path
import numpy as np

def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

