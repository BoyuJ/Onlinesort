from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_sessionstart(session):
    import torch
    torch.set_num_threads(1)
