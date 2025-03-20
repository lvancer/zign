from PIL import Image
from torchvision import transforms
from zign.data import zDataset, DatasetType

from config import zConfig

from torchvision import transforms
import numpy as np
import torch
from zign.utils.vo import transforms as ztransforms
from zign.utils.vo import Palette


class SegmentDataset(zDataset):
    
    def __init__(self, config: zConfig, dataset_type: DatasetType=DatasetType.TRAIN):
        super().__init__()
        self.config = config
        self.dataset_type = dataset_type
        self.file_pairs = self.create_file_pairs()
        self.palette = Palette(self.palette_array())
        
    def get_palette(self):
        return self.palette

    def palette_array(self):
        raise NotImplementedError

    def ignore_index(self):
        return 255
        
    def num_classes(self):
        raise NotImplementedError
        
    def create_file_pairs(self):
        raise NotImplementedError

    def load(self, index):
        if self.dataset_type.is_test():
            return [Image.open(self.file_pairs[index][0]), None]
        return [Image.open(path) for path in self.file_pairs[index]]

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, index):
        image, label = self.load(index)
        image = transforms.ToTensor()(image)
        
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        if label is None:
            return normalize(image)
        
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long().unsqueeze(0)
        if self.dataset_type.is_train():
            randomResizedCrop = ztransforms.RandomResizedCrop((self.config.image_size, self.config.image_size), (0.8, 1), antialias=True)
            randomResizedCrop.get_params(image)
            image_transforms = transforms.Compose([
                randomResizedCrop, 
                normalize
            ])
            label_transforms = randomResizedCrop.copy(interpolation=transforms.InterpolationMode.NEAREST, antialias=True)

        else:
            image_transforms = transforms.Compose([
                transforms.CenterCrop((self.config.image_size, self.config.image_size)),
                normalize
            ])
            label_transforms = transforms.Compose([
                transforms.CenterCrop((self.config.image_size, self.config.image_size))
            ])

        return image_transforms(image), label_transforms(label)            
