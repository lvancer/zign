from typing import List
from torchvision import transforms
from PIL import Image
from torchvision import transforms
import numpy as np
import torch


class Palette:
    
    def __init__(self, palette: List[int]):
        self.palette = palette
        
    def colorize_mask(self, mask: np.ndarray):
        zero_pad = 256 * 3 - len(self.palette)
        for i in range(zero_pad):
            self.palette.append(0)
        new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
        new_mask.putpalette(self.palette)
        return new_mask
    
    def colorize_tensor(self, label: torch.Tensor, index=0):
        _label = label.cpu()
        if len(label.shape) == 4:
            _label = _label.squeeze(1)
        elif len(label.shape) == 2:
            _label = _label.unsqueeze(0)
        l = np.asarray(_label.data[index].numpy(), dtype=np.uint8)
        mask = self.colorize_mask(l)
        return transforms.ToTensor()(mask.convert("RGB"))
    
    
def get_simple_palette(num_classes):
    n = num_classes
    palette = [0]*(n*3)
    for j in range(0,n):
            lab = j
            palette[j*3+0] = 0
            palette[j*3+1] = 0
            palette[j*3+2] = 0
            i = 0
            while (lab > 0):
                    palette[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                    palette[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                    palette[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                    i = i + 1
                    lab >>= 3
    return palette