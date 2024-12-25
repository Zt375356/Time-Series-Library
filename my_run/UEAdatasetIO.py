from models import TimesNet
import torch
import torch.nn as nn
import numpy as np
from exp.exp_classification import Exp_Classification
from data_provider.data_factory import data_provider
import argparse


parser = argparse.ArgumentParser()
args = parser.parse_args()

args.dataset_name = 'TEST'
args.task_name = 'classification'
args.data = 'UEA'
args.root_path = './dataset/'
args.batch_size = 16
args.seq_len = 100
args.num_workers = 0

train_data_set, train_data_loader = data_provider(args, flag='TRAIN')
test_data_set, test_data_loader = data_provider(args, flag='TEST')

