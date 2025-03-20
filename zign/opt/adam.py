import itertools
from typing import Union
import torch
from config import zConfig
from zign.opt.misc import _parameter


def _adam(optim, config: zConfig, **kwargs):
    def __adam(*params: Union[torch.nn.Parameter, torch.nn.Module]):
        parameters = [_parameter(param) for param in params]
        _params = itertools.chain(*parameters) if len(parameters) > 1 else  parameters[0]
        _kwargs = {
            'eps': config.opt_adam_epsilon,
            'weight_decay': config.opt_adam_weight_decay,
            'betas': (config.opt_adam_beta1, config.opt_adam_beta2),
        }
        _kwargs.update(kwargs)
        return optim(_params, lr=config.lr, **_kwargs)
    return __adam

def adam(config: zConfig, **kwargs):
    return _adam(torch.optim.Adam, config, **kwargs)


def adamW(config: zConfig, **kwargs):
    return _adam(torch.optim.AdamW, config, **kwargs)