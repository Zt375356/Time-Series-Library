import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append(r'c:\\Users\\W\\Desktop\\Time-Series-Library')

from exp.exp_classification import Exp_Classification
from data_provider.data_factory import data_provider
import argparse

dataset_names = [
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
]


parser = argparse.ArgumentParser()
args = parser.parse_args()

args.dataset_name = dataset_names[0]


args.task_name = 'classification'
args.data = 'UEA'
args.root_path = f'./dataset/{args.dataset_name}'
args.embed = 'timeF'
args.freq = None
args.batch_size = 16
args.num_workers = 0
args.augmentation_ratio = 0
args.seq_len = 100

train_data_set, train_data_loader = data_provider(args, flag='TRAIN')
test_data_set, test_data_loader = data_provider(args, flag='TEST')

for idx,batch in enumerate(train_data_loader):
    print(idx)
    print(batch[-1])
