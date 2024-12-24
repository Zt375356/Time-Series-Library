import torch
from torch import nn
from torch.nn import functional as F
import os
os.chdir(r'c:\\Users\\W\\Desktop\\Time-Series-Library')
import numpy as np

from args import args
import dataset_loader
import math
import tqdm
import random

import sys
sys.path.append(r"c:\\Users\\W\\Desktop\\Time-Series-Library")  # 将高一级目录添加到模块搜索路径
from process import Trainer,RayTrainer
from models import TimesNet, PatchTST 


import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch


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


def prepare_data(args):

    
    # 加载数据集
    print("正在加载数据集...")
    seq_len, num_classes, num_channel, train_loader, val_loader, test_loader = dataset_loader.load_data(args)

    print(f"数据集名称: {args.dataset_name}")
    print(f"数据形状: ({seq_len},{num_channel})")

    print("数据集加载完成!")
    return seq_len, num_classes, num_channel, train_loader, val_loader, test_loader


def main():

    # 初始化模型
    print("正在初始化模型...")
    model = PatchTST.Model(args).to(args.device)  # 将模型移动到指定设备
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    print("模型初始化完成!")
    print("="*50)

    # 初始化训练器
    print("正在初始化训练器...")
    sys.path.append(r"c:\\Users\\W\\Desktop\\Time-Series-Library\\my_run")  # 将高一级目录添加到模块搜索路径
    trainer = Trainer(args, model, train_loader, test_loader, verbose=True)
    print("训练器初始化完成!")
    print("="*50)

    # 设置训练模式
    if args.is_training:
        # 开始训练
        print("开始训练模型...")
        best_metric,current_loss = trainer.train()
        
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


def train_tune(config):
    # 更新args字典中的值以匹配config字典中的值
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)    
    # print(args)

    # 初始化模型    
    print("正在初始化模型...")
    model = PatchTST.Model(args).to(args.device)  # 将模型移动到指定设备
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    print("模型初始化完成!")
    print("="*50)

    # 初始化训练器
    print("正在初始化训练器...")
    sys.path.append(r"c:\\Users\\W\\Desktop\\Time-Series-Library\\my_run")  # 将高一级目录添加到模块搜索路径
    from process import RayTrainer
    trainer = RayTrainer(args, model, train_loader, test_loader, verbose=True)
    print("训练器初始化完成!")
    print("="*50)

    # 设置训练模式
    if args.is_training:
        # 开始训练
        print("开始训练模型...")
        best_metric,current_loss = trainer.train()

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
    

def custom_trial_name_creator(trial):
    name = "my_trial"
    for key, value in trial.config.items():
        name += f"_{key}_{value}"
    return name

if __name__ == '__main__':
    # 设置随机种子
    print("="*50)
    print("正在设置随机种子...")
    seed_everything(seed=2024)
    print("随机种子设置完成!")
    print("="*50)
    
    seq_len, num_classes, num_channel, train_loader, val_loader, test_loader = prepare_data(args=args)
    args.seq_len = seq_len
    args.enc_in = num_channel
    args.pred_len = 0 # for classification
    args.num_class = num_classes
    args.label_len = 1

    print("="*50)

    ray.init()
    result = tune.run(
        train_tune,
        config={
            "lr": tune.grid_search([0.0001]),
            "e_layers": tune.grid_search([2, 3, 4, 5]),
            "d_model": tune.grid_search([16, 32, 64, 128]),
            "d_ff": tune.grid_search([16, 32, 64, 128]),
            # "embed": tune.grid_search(['fixed', 'timeF']),
            # "freq": tune.grid_search(['h', 'd']),
            "top_k": tune.grid_search([3, 4, 5]),
            # "num_kernels": tune.grid_search([3]),
            # "dropout": tune.grid_search([0.1, 0.2, 0.3]),
            # "factor": tune.grid_search([1, 2, 3]),
            # "n_heads": tune.grid_search([8]),
            # "activation": tune.grid_search(['gelu', 'relu'])
        },
        search_alg=OptunaSearch(),
        resources_per_trial={"cpu": 8, "gpu": 1},
        num_samples=2,
        scheduler=ASHAScheduler(metric="accuracy", mode="max", grace_period=args.num_epoch//10),
        progress_reporter=CLIReporter(metric_columns=["accuracy", "training_iteration"]),
        checkpoint_at_end=False,
        storage_path=r"file:///C:/Users/W/Desktop/Time-Series-Library/logs/HAR-B",
        trial_name_creator=custom_trial_name_creator,
        trial_dirname_creator=custom_trial_name_creator,
    )
     # 找出最佳实验
    best_trial = result.get_best_trial("accuracy", "max", "last")
    # 打印最佳实验的参数配置
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


"""
HAR-B:PatchTST-Bacth_32s
Best trial config: {'lr': 0.0001, 'e_layers': 6, 'd_model': 128, 'd_ff': 512}
Best trial final validation loss: 0.011632021445095023
Best trial final validation accuracy: 0.8912489379779099
"""
