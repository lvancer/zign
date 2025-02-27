# class zLogger():
    
#     def __init__(self, handlers=[logging.StreamHandler(sys.stdout)]):
        
#         formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
#         logging.basicConfig(level=logging.DEBUG,
#                             format=formatter,
#                             datefmt='%Y-%m-%d %H:%M:%S',
#                             handlers=handlers)

import logging
from datetime import datetime
import os

class zLogger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(zLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, log_dir='.logs', log_name_prefix='zign'):
        if self._initialized:
            return
        self.log_dir = log_dir
        self.log_name_prefix = log_name_prefix
        self.logger = logging.getLogger(self.log_name_prefix)
        self.logger.setLevel(logging.DEBUG)  # 设置最低的日志级别为DEBUG
        self._initialized = True
        # 确保日志目录存在
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.setup_default_handlers()

    def setup_default_handlers(self):
        """设置默认的日志处理器"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f'{self.log_dir}/{self.log_name_prefix}_{timestamp}.log'
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        """获取已经配置好的logger对象"""
        return self.logger

    def add_file_handler(self, filename=None, level=logging.DEBUG):
        """动态添加文件处理器"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'{self.log_dir}/{self.log_name_prefix}_extra_{timestamp}.log'
        handler = logging.FileHandler(filename)
        handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)




        