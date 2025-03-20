from typing import Optional, Sequence, Union
import warnings
from torch import nn
from torchvision import transforms
from torchvision.utils import _log_api_usage_once
import torchvision.transforms.functional as F

class RandomResizedCrop(nn.Module):
    def __init__(
            self,
            size,
            scale=(0.08, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias: Optional[Union[str, bool]] = "warn",
            params=None,
        ):
            super().__init__()
            _log_api_usage_once(self)
            self.size = size

            if not isinstance(scale, Sequence):
                raise TypeError("Scale should be a sequence")
            if not isinstance(ratio, Sequence):
                raise TypeError("Ratio should be a sequence")
            if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
                warnings.warn("Scale and ratio should be of kind (min, max)")

            if isinstance(interpolation, int):
                interpolation = F._interpolation_modes_from_int(interpolation)

            self.interpolation = interpolation
            self.antialias = antialias
            self.scale = scale
            self.ratio = ratio
            self.params = None

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        params = self.get_params(img)
        return F.resized_crop(img, *params, self.size, self.interpolation, antialias=self.antialias)
    
    def get_params(self, img):
        if img is None:
            return self.params
        if self.params is not None:
            return self.params
        i, j, h, w = transforms.RandomResizedCrop.get_params(img, self.scale, self.ratio)
        self.params = (i, j, h, w)
        return self.params
    
    def copy(self, interpolation=None, antialias=None):
        return RandomResizedCrop(self.size, self.scale, self.ratio, 
                                  interpolation if interpolation is not None else self.interpolation, 
                                  antialias if antialias is not None else self.antialias, 
                                  self.params)