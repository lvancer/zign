from config import zConfig
from torch.optim import lr_scheduler

def step(config: zConfig):
    """
    每隔指定的步数（即 step_size的倍数），将学习率乘以一个固定的因子（即 gamma）
    """
    def _step(optimizer, **kwargs):
        if config.lr_step_size <= 0:
            assert False, 'lr_step_size must be greater than 0'
        _kwargs = {
            'step_size': config.lr_step_size,
            'gamma': config.lr_step_gamma,
            'verbose': config.lr_verbose,
        }
        _kwargs.update(kwargs)
        return lr_scheduler.StepLR(optimizer, **_kwargs)
    return _step