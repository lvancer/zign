from config import zConfig
from torch.optim import lr_scheduler

def cosine(config: zConfig):
    """
    基于余弦函数的特性逐步调整学习率。它允许学习率在训练过程中逐渐减小，随后将在训练的后期迅速降低，以便在收敛时保持稳定的更新
    """
    def _step(optimizer, **kwargs):
        if config.lr_cosine_T_max <= 0:
            assert False, 'lr_cosine_T_max must be greater than 0'
        _kwargs = {
            'T_max': config.lr_cosine_T_max,
            'eta_min': config.lr_cosine_eta_min,
            'verbose': config.lr_verbose,
        }
        _kwargs.update(kwargs)
        return lr_scheduler.CosineAnnealingLR(optimizer, **_kwargs)
    return _step