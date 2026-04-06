import os
import numpy as np

class Logger:
    """
    日志记录器
    - 记录训练过程中的各种指标
    - 支持 scalar 类型的日志
    """

    def __init__(self, log_dir):
        """
        初始化日志记录器
        :param log_dir: 日志目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 日志数据
        self.logs = {}

    def log_scalar(self, name, value, step):
        """
        记录标量值
        :param name: 指标名称
        :param value: 指标值
        :param step: 步骤
        """
        if name not in self.logs:
            self.logs[name] = []
        self.logs[name].append((step, value))

    def save_logs(self):
        """
        保存日志到文件
        """
        for name, data in self.logs.items():
            file_path = os.path.join(self.log_dir, f'{name}.npy')
            np.save(file_path, np.array(data))
            print(f"日志已保存到 {file_path}")

    def load_logs(self):
        """
        加载日志从文件
        """
        for file in os.listdir(self.log_dir):
            if file.endswith('.npy'):
                name = file[:-4]
                file_path = os.path.join(self.log_dir, file)
                data = np.load(file_path)
                self.logs[name] = data.tolist()
                print(f"日志已加载从 {file_path}")

    def get_log(self, name):
        """
        获取日志数据
        :param name: 指标名称
        :return: 日志数据
        """
        if name in self.logs:
            return self.logs[name]
        return []