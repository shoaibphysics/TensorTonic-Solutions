import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, data: list, label_col: int):
        self.data = data
        self.label_col = label_col

    def __len__(self) -> int:
        """
        Returns the number of rows.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (features, label) as float32 tensors of shapes (D,) and (1,).
        """
        row = torch.tensor(self.data[idx], dtype = torch.float32)
        
        label = row[self.label_col].reshape(1)
        mask = torch.ones(len(row), dtype = torch.bool)
        mask[self.label_col] = False
        return row[mask], label
