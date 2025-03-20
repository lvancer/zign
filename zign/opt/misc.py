from typing import Union
import torch


def _parameter(param: Union[torch.nn.Parameter, torch.nn.Module]):
    if isinstance(param, torch.nn.Module):
        return param.parameters()
    return param