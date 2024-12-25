import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvNetModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(ConvNetModel, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        # 第一组Conv1D和激活函数
        self.conv1 = nn.Conv1d(in_channels=input_shape[0], out_channels=100, kernel_size=10, stride=1)
        self.relu1 = nn.ReLU()

        # 第二组Conv1D和激活函数
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=10, stride=1)
        self.relu2 = nn.ReLU()

        # 最大池化层
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=3)

        # 第三组Conv1D和激活函数
        self.conv3 = nn.Conv1d(in_channels=100, out_channels=160, kernel_size=10, stride=1)
        self.relu3 = nn.ReLU()

        # 第四组Conv1D和激活函数
        self.conv4 = nn.Conv1d(in_channels=160, out_channels=160, kernel_size=10, stride=1)
        self.relu4 = nn.ReLU()

        # 全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Dropout层
        self.dropout = nn.Dropout(p=0.5)

        # 全连接层
        self.fc = nn.Linear(160, num_classes)

    def forward(self, x):
        x = x.permute(0,2,1)
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
