import sys
import os
import time


# 控制台输出记录到文件
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a",encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def log(log_path):
        # 自定义目录存放日志文件
        #log_path = './logdata/'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        # 日志文件名按照程序运行时间设置
        log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
        # 记录正常的 print 信息
        sys.stdout = Logger(log_file_name)
        # 记录 traceback 异常信息
        sys.stderr = Logger(log_file_name)



