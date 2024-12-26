import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvNetModel(nn.Module):
    def __init__(self, args):
        super(ConvNetModel, self).__init__()

        self.args = args
        self.num_channel = args.enc_in
        self.d_model = args.d_model
        self.num_class = args.num_class
        self.mlp_dim = args.d_ff
        self.dropout = args.dropout

        # 第一组Conv1D和激活函数
        self.conv1 = nn.Conv1d(in_channels=self.num_channel, out_channels=self.d_model, kernel_size=10, stride=1)
        self.relu1 = nn.ReLU()

        # 第二组Conv1D和激活函数
        self.conv2 = nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=10, stride=1)
        self.relu2 = nn.ReLU()

        # 最大池化层
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=3)

        # 第三组Conv1D和激活函数
        self.conv3 = nn.Conv1d(in_channels=self.d_model, out_channels=self.mlp_dim, kernel_size=10, stride=1)
        self.relu3 = nn.ReLU()

        # 第四组Conv1D和激活函数
        self.conv4 = nn.Conv1d(in_channels=self.mlp_dim, out_channels=self.mlp_dim, kernel_size=10, stride=1)
        self.relu4 = nn.ReLU()

        # 全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Dropout层
        self.dropout = nn.Dropout(self.dropout)

        # 全连接层
        self.fc = nn.Linear(self.mlp_dim, self.num_class)


    def classification(self, x, x_mark_enc):

        # 第一组卷积和激活函数
        x = self.relu1(self.conv1(x))

        # 第二组卷积和激活函数
        x = self.relu2(self.conv2(x))

        # 最大池化
        x = self.maxpool(x)

        # 第三组卷积和激活函数
        x = self.relu3(self.conv3(x))

        # 第四组卷积和激活函数
        x = self.relu4(self.conv4(x))

        # 全局平均池化
        x = self.global_avg_pool(x)

        # 展平
        x = x.view(x.size(0), -1)

        # Dropout
        x = self.dropout(x)

        # 全连接层
        x = self.fc(x)
        return x


    def forward(self, x, x_mark_enc):
        out = self.classification(x,x_mark_enc)
        return out
      
