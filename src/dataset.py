import copy
from torch.utils.data import Dataset, DataLoader


class MusicDataset(Dataset):
    def __init__(self, data_list, transform=None):
        super(MusicDataset, self).__init__()
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        data = copy.deepcopy(self.data_list[i])
        return self.transform(data) if self.transform is not None else data

    def get_loader(self, batch_size=1, shuffle=False, num_workers=0):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
