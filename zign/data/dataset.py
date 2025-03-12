from typing import Callable, List, Any
from zign.data.abc import BaseDataset
from torch.utils import data
from typing import TypeVar, Generic, Tuple, Optional


class zDataset(BaseDataset):
    
    collate_fn = None
    
    def dataloader(self, batch_size, shuffle=True, collate_fn=None, *args, **kwargs)-> data.DataLoader:
        if collate_fn is None and self.collate_fn is not None:
            collate_fn = self.collate_fn
        return data.DataLoader(self, batch_size, shuffle, collate_fn=collate_fn, *args, **kwargs)
    
    def set_collate_fn(self, collate_fn: Callable[[List[Any]], Any]):
        self.collate_fn = collate_fn