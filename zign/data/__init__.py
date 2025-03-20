from .dataset import zDataset
from enum import Enum

class DatasetType(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3
    
    def is_train(self):
        return self == DatasetType.TRAIN
    
    def is_val(self):
        return self == DatasetType.VAL
    
    def is_test(self):
        return self == DatasetType.TEST