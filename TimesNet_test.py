from models import TimesNet
import torch
import torch.nn as nn
import numpy as np

class Configs:
    def __init__(self):
        self.task_name = 'classification'
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 96
        self.e_layers = 3  # 根据file_context_0修改为3
        self.d_model = 128
        self.d_ff = 256  # 根据file_context_0添加缺失的属性
        self.c_out = 7
        self.enc_in = 7
        self.embed = 'timeF'
        self.freq = 'h'
        self.dropout = 0.05
        self.num_class = 2
        self.top_k = 3
        self.num_kernels = 3  # 补充该缺少属性

model = TimesNet.TimesBlock(Configs())


class TestData:
    def __init__(self):
        # 序列长度
        self.seq_len = 96
        # 预测长度
        self.pred_len = 96
        # 模型维度
        self.d_model = 128
        # 前馈网络维度
        self.d_ff = 256
        # 输出通道数
        self.c_out = 7
        # 编码器输入维度
        self.enc_in = 7
        # 嵌入类型
        self.embed = 'timeF'
        # 频率类型
        self.freq = 'h'
        # dropout率
        self.dropout = 0.05
        # 类别数量
        self.num_class = 2
        # 选择的top k个频率
        self.top_k = 3
        # 卷积核数量
        self.num_kernels = 3

    def generate_data(self):
        x = np.random.rand(64, self.seq_len, 3)
        x = torch.tensor(x, dtype=torch.float32)
        return x

    def test_model(self, model):
        x = self.generate_data()
        output = model(x)
        print(output.shape)

test = TestData().test_model(model)

