from torch.utils.data import Dataset


class exp_dataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]
