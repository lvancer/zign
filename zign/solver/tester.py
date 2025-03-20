from typing import TypeVar, Generic, Optional
from zign.config import zConfig
import torch
import logging


Co = TypeVar('Co', bound=zConfig)

class zTester(Generic[Co]):

    def __init__(self, config: Optional[Co]):
        self.config = config
        
    def test_one(self, idx, inputs):
        pass
        
    def test(self, dataset, batch_size=1, shuffle=False):
        dataloader = dataset.dataloader(batch_size, shuffle)
        for idx, inputs in enumerate(dataloader):
            with torch.no_grad():
                self.test_one(idx, inputs)
                logging.info(f'{idx+1}/{len(dataloader)}')
                

