from abc import ABC, abstractmethod
from torch.utils import data


class BaseDataset(data.Dataset):
    
    @abstractmethod
    def __len__(self):
        return 0
    
    @abstractmethod
    def __getitem__(self, index):
        pass

    
    
        