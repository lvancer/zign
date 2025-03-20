from config import zConfig
from torch.optim import lr_scheduler

def linear(config: zConfig):
    def _linear(optimizer, **kwargs):
        def lambda_rule(epoch):
            if config.lr_linear_decay <= 0:
                assert False, 'lr_linear_decay must be greater than 0'
            lr_l = 1.0 - max(0, epoch + 1 - (config.num_epochs - config.lr_linear_decay)) / float(config.lr_linear_decay + 1)
            return lr_l
        _kwargs = {
            'lr_lambda': lambda_rule,
            'verbose': config.lr_verbose,
        }
        _kwargs.update(kwargs)
        return lr_scheduler.LambdaLR(optimizer, **_kwargs)
        
    return _linear


