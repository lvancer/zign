from typing import TypeVar, Generic, Optional
from zign.config import zConfig
import torch


Co = TypeVar('Co', bound=zConfig)

class zTester(Generic[Co]):

    def __init__(self, config: Optional[Co]):
        self.config = config
        
    def forward(self, idx, inputs):
        pass
    
    def eval(self, idx, inputs, outputs, dataset):
        pass
        
    def test(self, dataset):
        dataloader = dataset.dataloader(1, False)
        for idx, inputs in enumerate(dataloader):
            with torch.no_grad():
                outputs = self.forward(idx, inputs)
                self.eval(idx, inputs, outputs, dataset)