from typing import Callable, List, Any
from zign.data.abc import BaseDataset
from torch.utils import data
from typing import TypeVar, Generic, Tuple, Optional
from zign.data.sampler import EpochSplitSampler

class zDataset(BaseDataset):
    
    def __init__(self, sample_ratio=1, cache=False):
        self.epoch_count = 0
        self.sample_ratio = sample_ratio
        self._sampler = None
        self._collate_fn = None
        self._split = 0
        self._test = False
        self.cache = cache
    
    def dataloader(self, batch_size, shuffle=True, collate_fn=None, sampler=None, *args, **kwargs)-> data.DataLoader:
        if collate_fn is None and self._collate_fn is not None:
            collate_fn = self._collate_fn
        if sampler is None and self._sampler is not None:
            sampler = self._sampler
            
        if self._split > 1:
            sampler = EpochSplitSampler(len(self), self._split, self.epoch_count % self._split, shuffle)
            return data.DataLoader(self, batch_size, shuffle=False, collate_fn=collate_fn, sampler=sampler, *args, **kwargs)
        
        return data.DataLoader(self, batch_size, shuffle, collate_fn=collate_fn, sampler=sampler, *args, **kwargs)
    
    def set_collate_fn(self, collate_fn: Callable[[List[Any]], Any]):
        self._collate_fn = collate_fn
        
    def set_sampler(self, sampler: data.Sampler):
        self._sampler = sampler
    
    def split(self, split):
        self._split = split
        return self
    
    def test(self):
        self._test = True
        return self
    
    def is_test(self):
        return self._test