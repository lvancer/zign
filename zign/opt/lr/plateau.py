from torch.optim import lr_scheduler
from config import zConfig

def plateau(config: zConfig):
    """
    当模型的性能(如验证损失)在经过一段时间(指定的patience)后未见提高(threshold)时，降低学习率
    """
    def _plateau(optimizer, **kwargs):
        _kwargs = {
            'mode': config.lr_plateau_mode,
            'factor': config.lr_plateau_factor,
            'threshold': config.lr_plateau_threshold,
            'patience': config.lr_plateau_patience,
            'verbose': config.lr_verbose,
        }
        _kwargs.update(kwargs)
        return lr_scheduler.ReduceLROnPlateau(optimizer, **_kwargs)
    return _plateau



