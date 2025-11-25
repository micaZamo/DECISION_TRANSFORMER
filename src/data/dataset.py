import numpy as np
import torch
from torch.utils.data import Dataset

class RecommendationDataset(Dataset):
    """
    Dataset para Decision Transformer.
    Cada ejemplo es una ventana de longitud fija (context_length)
    sacada de una trayectoria de un usuario.
    """
    def __init__(self, trajectories, context_length=20):
        self.trajectories = trajectories
        self.context_length = context_length

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]

        items = traj["items"]
        ratings = traj["ratings"]
        rtg = traj["returns_to_go"]
        timesteps = traj["timesteps"]
        group = traj["user_group"]

        # ventana fija
        seq_len = min(len(items), self.context_length)

        if len(items) > self.context_length:
            start = np.random.randint(0, len(items) - self.context_length + 1)
        else:
            start = 0

        end = start + seq_len

        states = items[start:end]
        actions = items[start:end]

        targets = np.empty(seq_len, dtype=np.int64)
        targets[:-1] = items[start+1:end]
        targets[-1] = -1  # padding para último timestep

        rtg_seq = rtg[start:end].reshape(-1, 1)
        time_seq = timesteps[start:end]

        return {
            "states": torch.tensor(states, dtype=torch.long),
            "actions": torch.tensor(actions, dtype=torch.long),
            "rtg": torch.tensor(rtg_seq, dtype=torch.float32),
            "timesteps": torch.tensor(time_seq, dtype=torch.long),
            "groups": torch.tensor(group, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
        }
