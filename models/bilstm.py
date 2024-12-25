import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=2, dropout_prob=0.1, device="cuda0"):
        super(BiLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.device = device
        # 定义前向传播层
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_prob,
                            bidirectional=True)

        # 定义Dropout层
        self.dropout = nn.Dropout(dropout_prob)

        # 定义全连接层
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # 双向LSTM的hidden_dim * 2

    def forward(self, x):
        # 确保x的形状是[batch_size, sequence_length, input_dim]
        batch_size = x.size(0)

        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).requires_grad_().to(self.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).requires_grad_().to(self.device)

        # 前向传播LSTM层
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出用于分类
        out = self.dropout(out[:, -1, :])

        # 全连接层
        out = self.fc(out)

        return out
