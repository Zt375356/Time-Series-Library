from models import TimesNet
import torch
import torch.nn as nn
import numpy as np

class Configs:
    def __init__(self):
        self.task_name = 'classification'
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 0
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


model = TimesNet.Model(Configs())


class TestData:
    def __init__(self):

        self.x_mark_enc = torch.tensor(np.ones((64, 96, 128)), dtype=torch.float32)  # 修改x_mark_enc为随机生成的测试数据
        self.x_dec = torch.tensor(np.random.rand(64, 96, 7), dtype=torch.float32)  # 修改x_dec为随机生成的测试数据

    def generate_data(self):
        x = np.random.rand(64, 96, 7)
        x = torch.tensor(x, dtype=torch.float32)
        return x

    def test_model(self, model):
        x = self.generate_data()
        output = model(x,self.x_mark_enc,None,None)
        print(output.shape)


test = TestData().test_model(model)
