from .logger import zLogger

# 创建全局的CustomLogger实例并提供便捷函数
_logger = zLogger(log_dir='.logs', log_name_prefix='zign')

def debug(msg, *args, **kwargs):
    _logger.get_logger().debug(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    _logger.get_logger().info(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    _logger.get_logger().warning(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    _logger.get_logger().error(msg, *args, **kwargs)

def critical(msg, *args, **kwargs):
    _logger.get_logger().critical(msg, *args, **kwargs)











