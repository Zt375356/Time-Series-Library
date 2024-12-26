from regex import P
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
from data_provider.data_factory import data_provider
from process import Trainer,RayTrainer,WandBTrainer
import models

import wandb


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

    """
    UEA:[batch_size,seq_len,num_channel]
    Private:[batch_size,num_channel,seq_len]
    
    """
    dataset_names = {
        'UEA': [
            "EthanolConcentration",
            "FaceDetection",
            "Handwriting",
            "Heartbeat",
            "JapaneseVowels",
            "PEMS-SF",
            "SelfRegulationSCP1",
            "SelfRegulationSCP2",
            "SpokenArabicDigits",
            "UWaveGestureLibrary"
        ],
        'Private': [
            'AW-A',
            'AW-B',
            'Gesture-A',
            'Gesture-B',
            'HAR-A',
            'HAR-B',
            'HAR-C'
        ]
    }

    dataset_type = None
    for dataset_type, names in dataset_names.items():
        if args.dataset_name in names:
            break
    else:
        raise ValueError(f"Dataset name {args.dataset_name} not found in the dataset names.")

    if dataset_type == 'UEA':
        print(f"正在加载UEA数据集:{args.dataset_name}...")
        args.data = 'UEA'
        args.root_path = f'./dataset/{args.dataset_name}'
        train_data_set, train_loader = data_provider(args, flag='TRAIN')
        test_data_set, test_loader = data_provider(args, flag='TEST')
        seq_len, num_channel = train_data_set[0][0].size()
        num_classes = len(set(item[1] for item in train_data_set))
        dataset_length = len(train_data_set)
        print(f"数据集名称: {args.dataset_name}")
        print(f"数据形状: ({dataset_length},{seq_len},{num_channel})")
        print("数据集加载完成!")
        return seq_len, num_classes, num_channel, train_loader, test_loader, test_loader
    elif dataset_type == 'Private':
        print(f"正在加载Private数据集:{args.dataset_name}...")
        args.data = 'Private'
        seq_len, num_classes, num_channel, train_loader, val_loader, test_loader = dataset_loader.load_data(args)
        print(f"数据集名称: {args.dataset_name}")
        print(f"数据形状: ({seq_len},{num_channel})")
        print("数据集加载完成!")
        return seq_len, num_classes, num_channel, train_loader, val_loader, test_loader


def train_wandb():
    wandb.init(project=f"{args.dataset_name}", entity=f"{args.model}-{args.dataset_name}")
    for key in wandb.config.keys():
        setattr(args, key, wandb.config[key])

    seq_len, num_class, num_channel, train_loader, val_loader, test_loader = prepare_data(args=args)
    args.seq_len = seq_len
    args.enc_in = num_channel
    args.num_class = num_class

    # 初始化模型    
    print("正在初始化模型...")
    model = models.TSTformer(args).to(args.device)  # 将模型移动到指定设备
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    print("模型初始化完成!")
    print("="*50)

    # 初始化训练器
    print("正在初始化训练器...")

    trainer = WandBTrainer(args, model, train_loader, test_loader, verbose=True)
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
    

if __name__ == '__main__':
    # 设置随机种子
    print("="*50)
    print("正在设置随机种子...")
    seed_everything(seed=2024)
    print("随机种子设置完成!")
    print("="*50)

    args.pred_len = 0 # for classification
    args.label_len = 1

    # 使用wandb的sweep功能进行超参数调整
    sweep_config = {
        "name": f"{args.model}-{args.dataset_name}-sweep",
        "method": "bayes",
        'metric': {
        'goal': 'maximize', 
        'name': 'test/accuracy'
        },
        "parameters": {
            "lr":{"min": 0.0001, "max": 0.001},
            "batch_size":{"values": [16,32]},
            "e_layers": {"values": [3]},
            "d_model": {"values": [16, 32]},
            "d_ff": {"values": [32, 64,128]},
            # "embed": {"values": ['fixed', 'timeF']},
            # "top_k": {"values": [3]},
            # "dropout": {"values": [0.1,0.2]},
        },
        "early_terminate": {
        "type": "hyperband",
        "min_iter": 3
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=f"{args.model}-{args.dataset_name}")
    wandb.agent(sweep_id, train_wandb)