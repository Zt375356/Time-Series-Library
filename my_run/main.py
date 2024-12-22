import torch
from torch import nn
from torch.nn import functional as F
import os
import numpy as np

from args import args
import dataset_loader
from process import Trainer

from utils import *
import math
import tqdm
import random

import sys
sys.path.append("c:\\Users\\W\\Desktop\\Time-Series-Library")  # 将高一级目录添加到模块搜索路径
import models.TimesNet as TimesNet 

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def main():
    # 设置随机种子
    print("="*50)
    print("正在设置随机种子...")
    seed_everything(seed=2024)
    print("随机种子设置完成!")
    print("="*50)

    # 加载数据集
    print("正在加载数据集...")
    seq_len, num_classes, num_channel, train_loader, val_loader, test_loader = dataset_loader.load_data(args)

    print(f"数据集名称: {args.dataset_name}")
    args.seq_len = seq_len
    args.enc_in = num_channel
    args.pred_len = 0 # for classification
    args.num_class = num_classes
    args.label_len = 1
    print(f"数据形状: ({seq_len},{num_channel})")

    print("数据集加载完成!")
    print("="*50)

    # 初始化模型
    print("正在初始化模型...")
    model = TimesNet.Model(args).to(args.device)  # 将模型移动到指定设备

    print("模型初始化完成!")
    print("="*50)

    # 初始化训练器
    print("正在初始化训练器...")
    trainer = Trainer(args, model, train_loader, test_loader, verbose=True)
    print("训练器初始化完成!")
    print("="*50)

    # 设置训练模式
    if args.is_training:
        # 开始训练
        print("开始训练模型...")
        trainer.train()
    else:
        # 加载最佳模型
        print("正在加载最佳模型...")
        model.load_state_dict(torch.load(args.save_path + '/model.pkl'))
        print("最佳模型加载完成!")
        print("="*50)

        # 开始测试
        print("开始测试模型...")
        trainer.eval_model_vqvae()
        print("模型测试完成!")

if __name__ == '__main__':
    main()
