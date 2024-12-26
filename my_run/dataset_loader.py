import os
from glob import glob
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import re
import time
import subprocess

class MyDataset(Dataset):
    def __init__(self, datafolder, sequence_len, test_num=-1, type="train"):
        self.data = []
        self.labels = []

        for fi in os.listdir(datafolder):
            if fi.endswith(".npy") and fi[0] == 'X':
                xy, nn = fi.split('_')
                n, _ = nn.split('.')
                if (type == "train") and (int(n) != test_num):
                    for idx in range(155 // sequence_len):
                        start_idx = idx * sequence_len
                        end_idx = start_idx + sequence_len
                        if idx == (155 // sequence_len) - 1 and idx != 0:
                            pass
                        else:
                            x_data = np.load(os.path.join(datafolder, 'X_' + nn))[:, start_idx:end_idx, :]
                            y_data = np.load(os.path.join(datafolder, 'Y_' + nn))

                        # 确保x_data和y_data的数量一致
                        if x_data.shape[0] == y_data.shape[0]:
                            self.data.append(x_data)
                            self.labels.append(y_data)
                        else:
                            print(f"Warning: Mismatch in data and label counts for file {fi}")
                elif (type == "test") and (int(n) == test_num):
                    for idx in range(155 // sequence_len):
                        start_idx = idx * sequence_len
                        end_idx = start_idx + sequence_len
                        if idx == (155 // sequence_len) - 1 and idx != 0:
                            pass
                        else:
                            x_data = np.load(os.path.join(datafolder, 'X_' + nn))[:, start_idx:end_idx, :]
                            y_data = np.load(os.path.join(datafolder, 'Y_' + nn))

                        # 确保x_data和y_data的数量一致
                        if x_data.shape[0] == y_data.shape[0]:
                            self.data.append(x_data)
                            self.labels.append(y_data)
                        else:
                            print(f"Warning: Mismatch in data and label counts for file {fi}")

        self.data = np.vstack(self.data)
        self.labels = np.vstack(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
        # return x, self.labels[idx]


class NumpyDataset(Dataset):
    def __init__(self, data_file_path, labels_file_path, transform=None):
        self.data = np.load(data_file_path)
        self.labels = np.load(labels_file_path)  # -14for第二组实验
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label


class MultiDataset(Dataset):
    r"""
    A Dataset Class for building Dataloader of ECG or other datasets.
    """

    def __init__(
            self,
            samples,
            tokenizer,
            mode: str,
            multi: str,
            encoder_max_length=256,
            prefix_text="",
    ) -> None:
        assert mode in ["train", "test"]
        super().__init__()
        self.samples = samples
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = encoder_max_length
        self.multi = multi
        self.prefix_tokens = self.tokenizer.encode(prefix_text) if prefix_text else []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 从样本中提取文本、ECG信号和其他信息
        text, ecg, _ = self.samples[idx]

        # 寻找文本中"information.\n"的位置，以此作为诊断信息的开始
        dx_index = text.find("information.\n")
        if dx_index != -1:
            # 如果找到，则将文本分为诊断信息和其他文本
            label = text[dx_index + 13:]
            text = text[:dx_index + 13]
        else:
            # 如果没有找到，则诊断信息为空
            label = ''
        # 将诊断信息转换为标签ID
        label_ids = self.tokenizer.encode(label)

        # 根据模式决定是否将诊断信息添加到文本中
        if self.mode == "train":
            text = text + label
        else:
            text = text

        # 根据ECG信号和文本生成输入ID
        input_ids = self.template(ecg * 2.5, text)
        # 将标签ID填充到输入ID的长度，以便与输入ID对齐
        label_ids = [-100] * (len(input_ids) - len(label_ids)) + label_ids

        # 初始化注意力掩码，所有位置都为1
        attn_masks = [1] * len(input_ids)
        # 填充输入ID和注意力掩码以达到最大长度
        input_ids, attn_masks = self.pad(input_ids, attn_masks)
        # 填充标签ID以达到最大长度
        label_ids, _ = self.pad(label_ids, attn_masks)

        if self.mode == "train":
            return {
                "input_ids": torch.LongTensor(input_ids),
                "attention_mask": torch.FloatTensor(attn_masks),
                "label_ids": torch.LongTensor(label_ids),
            }

        elif self.mode == "test":
            return {
                "input_ids": torch.LongTensor(input_ids),
                "attn_masks": torch.FloatTensor(attn_masks),
                "label": label,
            }

    def template(self, ecg, text):
        r"""
        The contents of the items are stitched together according to a template to construct the input.
        """
        input_ids = self.prefix_tokens.copy()
        if self.multi == 'mix':
            if ecg.shape == (155, 6):  # AW-A、AW-B
                bet_ids = self.tokenizer.encode('Aerial gesture signals: <BET>')
                ecg_ids = self.tokenizer.encode(torch.Tensor(ecg).unsqueeze(0), model_id=0)
            elif ecg.shape == (250, 10):  # Gesture-A
                bet_ids = self.tokenizer.encode('Gesture signals: <BET>')
                ecg_ids = self.tokenizer.encode(torch.Tensor(ecg).unsqueeze(0), model_id=1)
            elif ecg.shape == (20, 6):  # Gesture-B
                bet_ids = self.tokenizer.encode('Gesture signals: <BET>')
                ecg_ids = self.tokenizer.encode(torch.Tensor(ecg).unsqueeze(0), model_id=2)
            elif ecg.shape == (128, 9):  # HAR-A
                bet_ids = self.tokenizer.encode('Human physical activities signals: <BET>')
                ecg_ids = self.tokenizer.encode(torch.Tensor(ecg).unsqueeze(0), model_id=3)
            elif ecg.shape == (151, 3):  # HAR-B
                bet_ids = self.tokenizer.encode('Human physical activities signals: <BET>')
                ecg_ids = self.tokenizer.encode(torch.Tensor(ecg).unsqueeze(0), model_id=4)
            elif ecg.shape == (256, 9):  # HAR-C
                bet_ids = self.tokenizer.encode('Human physical activities signals: <BET>')
                ecg_ids = self.tokenizer.encode(torch.Tensor(ecg).unsqueeze(0), model_id=5)
        else:
            if self.multi.startswith('AW'):
                bet_ids = self.tokenizer.encode('Aerial gesture signals: <BET>')
            elif self.multi.startswith('HAR'):
                bet_ids = self.tokenizer.encode('Human physical activities signals: <BET>')
            elif self.multi.startswith('Gesture'):
                bet_ids = self.tokenizer.encode('Gesture signals: <BET>')
                
            ecg_ids = self.tokenizer.encode(torch.Tensor(ecg).unsqueeze(0))
        text_ids = self.tokenizer.encode('<EET> \n' + text)

        ecg_ids = ecg_ids.tolist()
        ecg_ids = ecg_ids[0]

        input_ids.extend(bet_ids + ecg_ids + text_ids)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[0: self.max_length]
        
        return input_ids

    def pad(self, input_ids: list, attn_masks: list):
        r"""
        为GPT模型的输入进行填充。

        在训练模式下，我们在右侧进行填充，
        在测试模式下，我们在左侧进行填充。
        """
        assert len(input_ids) <= self.max_length

        # 根据当前的模式决定填充的方向
        if self.mode == "train":
            # 在训练模式下，我们在输入ID和注意力掩码的右侧进行填充
            # 使用tokenizer的填充标记符填充到最大长度
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (
                self.max_length - len(input_ids)
            )
            # 对应地，注意力掩码也需要进行填充，以确保长度一致
            # 填充的值为0，表示这些位置不需要关注
            attn_masks = attn_masks + [0] * (self.max_length - len(attn_masks))
        elif self.mode == "dev" or self.mode == "test":
            # 在测试或开发模式下，我们在输入ID和注意力掩码的左侧进行填充
            # 使用tokenizer的填充标记符填充到最大长度
            input_ids = [self.tokenizer.pad_token_id] * (
                self.max_length - len(input_ids)
            ) + input_ids
            # 对应地，注意力掩码也需要进行填充，以确保长度一致
            # 填充的值为0，表示这些位置不需要关注
            attn_masks = [0] * (self.max_length - len(attn_masks)) + attn_masks
        return input_ids, attn_masks


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type='standardization', mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


# 定义读取数据集的函数
def read_dataset(file_path, delimiter):
    """
    读取数据集并返回DataFrame的数值部分。

    参数:
    - file_path: 文件的路径（字符串）
    - delimiter: 文件中使用的分隔符（字符串）

    返回:
    - numpy数组: 包含文件数值的数组
    """
    # 读取数据集，假设第一列是标签，我们只取剩余的数值列
    df = pd.read_csv(file_path, delimiter=delimiter, engine='python', header=None, skiprows=1)
    return df.values  # 取从第二列到最后一列的数值


def HAR_dataset_loader(batch_size=128, sequence_len=128, id=1, use6dimension=False):
    """
    Human Activity Recognition Using Smartphones

    10,299 9-dimension samples from 6 daily activities
    Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra Perez, and Jorge Luis Reyes Ortiz. A public domain dataset for human activity recognition using smartphones. In ESANN, pages 437–442, 2013.
    return(batch_size,128,9)
    """

    if id == 1:
        # 定义数据集的路径
        dataset_paths_train = {
            'body_acc_x': r"C:\Users\W\Desktop\导入\数据集\HAR\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\train\Inertial Signals\body_acc_x_train.txt",
            'body_acc_y': r"C:\Users\W\Desktop\导入\数据集\HAR\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\train\Inertial Signals\body_acc_y_train.txt",
            'body_acc_z': r"C:\Users\W\Desktop\导入\数据集\HAR\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\train\Inertial Signals\body_acc_z_train.txt", 
            'body_gyro_x': r"C:\Users\W\Desktop\导入\数据集\HAR\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\train\Inertial Signals\body_gyro_x_train.txt",
            'body_gyro_y': r"C:\Users\W\Desktop\导入\数据集\HAR\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\train\Inertial Signals\body_gyro_y_train.txt",
            'body_gyro_z': r"C:\Users\W\Desktop\导入\数据集\HAR\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\train\Inertial Signals\body_gyro_z_train.txt",
            'total_acc_x': r"C:\Users\W\Desktop\导入\数据集\HAR\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\train\Inertial Signals\total_acc_x_train.txt",
            'total_acc_y': r"C:\Users\W\Desktop\导入\数据集\HAR\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\train\Inertial Signals\total_acc_y_train.txt",
            'total_acc_z': r"C:\Users\W\Desktop\导入\数据集\HAR\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\train\Inertial Signals\total_acc_z_train.txt",
        }
        # 定义数据集的路径
        dataset_paths_test = {
            'body_acc_x': r"C:\Users\W\Desktop\导入\数据集\HAR\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\test\Inertial Signals\body_acc_x_test.txt",
            'body_acc_y': r"C:\Users\W\Desktop\导入\数据集\HAR\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\test\Inertial Signals\body_acc_y_test.txt",
            'body_acc_z': r"C:\Users\W\Desktop\导入\数据集\HAR\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\test\Inertial Signals\body_acc_z_test.txt",
            'body_gyro_x': r"C:\Users\W\Desktop\导入\数据集\HAR\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\test\Inertial Signals\body_gyro_x_test.txt", 
            'body_gyro_y': r"C:\Users\W\Desktop\导入\数据集\HAR\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\test\Inertial Signals\body_gyro_y_test.txt",
            'body_gyro_z': r"C:\Users\W\Desktop\导入\数据集\HAR\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\test\Inertial Signals\body_gyro_z_test.txt",
            'total_acc_x': r"C:\Users\W\Desktop\导入\数据集\HAR\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\test\Inertial Signals\total_acc_x_test.txt",
            'total_acc_y': r"C:\Users\W\Desktop\导入\数据集\HAR\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\test\Inertial Signals\total_acc_y_test.txt",
            'total_acc_z': r"C:\Users\W\Desktop\导入\数据集\HAR\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\test\Inertial Signals\total_acc_z_test.txt",
        }

        dataset_label_paths_train = {
            'label': r"C:\Users\W\Desktop\导入\数据集\HAR\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\train\y_train.txt"
        }

        dataset_label_paths_test = {
            'label': r"C:\Users\W\Desktop\导入\数据集\HAR\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\test\y_test.txt"
        }

        delimiter = r'\s+'

        # 读取数据集并存储为numpy数组列表
        datasets_train = [read_dataset(path, delimiter) for path in dataset_paths_train.values()]
        datasets_test = [read_dataset(path, delimiter) for path in dataset_paths_test.values()]
        datasets_train_label = [read_dataset(path, delimiter) for path in dataset_label_paths_train.values()]
        datasets_test_label = [read_dataset(path, delimiter) for path in dataset_label_paths_test.values()]

        # 将列表中的numpy数组合并为一个9维数组
        train_data = np.dstack(datasets_train)
        test_data = np.dstack(datasets_test)
        train_label = np.array(datasets_train_label).squeeze(0).squeeze(-1)-1
        test_label = np.array(datasets_test_label).squeeze(0).squeeze(-1)-1

        X_train = torch.from_numpy(train_data).type(torch.float32)
        y_train = torch.from_numpy(train_label).type(torch.LongTensor)  # data eval_type is long

        # create feature and targets tensor for test set.
        X_test = torch.from_numpy(test_data).type(torch.float32)
        y_test = torch.from_numpy(test_label).type(torch.LongTensor)  # data eval_type is long

        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42, shuffle=True)

        # 进行norm化
        normalizer = Normalizer()
        X_train = normalizer.normalize(X_train)
        X_test = normalizer.normalize(X_test)
        X_val = normalizer.normalize(X_val)

        if use6dimension:
            train = torch.utils.data.TensorDataset(X_train[:,:,:6], y_train)
            test = torch.utils.data.TensorDataset(X_test[:,:,:6], y_test)
        else:
            train = torch.utils.data.TensorDataset(X_train, y_train)
            test = torch.utils.data.TensorDataset(X_test, y_test)
        # data loader
        val = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False) 
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    elif id == 2:
        dataset_path = r"C:\Users\W\Desktop\导入\emergency gesture re\开源数据集\UnimibShar-main\acc_data.csv"
        dataset_label_path = r"C:\Users\W\Desktop\导入\emergency gesture re\开源数据集\UnimibShar-main\acc_labels.csv"
        data = np.array(pd.read_csv(dataset_path))
        data_label = np.array(pd.read_csv(dataset_label_path))[:,0]-1
        data = data.reshape(data.shape[0], 3, 151).transpose(0, 2, 1)  # 453:是三个151，分别表示加速度计x，y，z的三个窗口

        X_train, X_temp, y_train, y_temp = train_test_split(data ,data_label, train_size=0.8, random_state=42, shuffle=True)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=0.5, random_state=42, shuffle=True)
        X_train = torch.from_numpy(X_train).to(torch.float32)
        y_train = torch.from_numpy(y_train).type(torch.LongTensor)  # data eval_type is long

        X_val = torch.from_numpy(X_val).to(torch.float32)
        y_val = torch.from_numpy(y_val).type(torch.LongTensor)  # data eval_type is

        # create feature and targets tensor for test set.
        X_test = torch.from_numpy(X_test).to(torch.float32)
        y_test = torch.from_numpy(y_test).type(torch.LongTensor)  # data eval_type is long

        # Pytorch train and test sets
        train = torch.utils.data.TensorDataset(X_train, y_train)
        val = torch.utils.data.TensorDataset(X_val, y_val)
        test = torch.utils.data.TensorDataset(X_test, y_test)

        # data loader
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=False)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False)

        return train_loader, val_loader, test_loader

    elif id == 3:
        """
        (2020)PTB-XL, a large publicly available electrocardiography dataset (version 1.0.1). PhysioNet.
        return [bz,256,9]
        """
        print(os.getcwd())
        X_train, y_train = np.load(r'C:\Users\admin\Desktop\emergency gesture re\开源数据集\HAR\har_train_all.npy'),\
                           np.load(r'C:\Users\admin\Desktop\emergency gesture re\开源数据集\HAR\har_train_label.npy')
        X_val, y_val = np.load(r'C:\Users\admin\Desktop\emergency gesture re\开源数据集\HAR\har_valid_all.npy'),\
                       np.load(r'C:\Users\admin\Desktop\emergency gesture re\开源数据集\HAR\har_valid_label.npy')
        X_test, y_test = np.load(r'C:\Users\admin\Desktop\emergency gesture re\开源数据集\HAR\har_test_all.npy'),\
                         np.load(r'C:\Users\admin\Desktop\emergency gesture re\开源数据集\HAR\har_test_label.npy')

        # Pytorch train and test sets
        X_train, y_train = torch.from_numpy(X_train).permute(0,2,1), torch.from_numpy(y_train)
        X_val, y_val = torch.from_numpy(X_val).permute(0,2,1), torch.from_numpy(y_val)
        X_test, y_test = torch.from_numpy(X_test).permute(0,2,1), torch.from_numpy(y_test)

        train = torch.utils.data.TensorDataset(X_train, y_train)
        val = torch.utils.data.TensorDataset(X_val, y_val)
        test = torch.utils.data.TensorDataset(X_test, y_test)

        # data loader
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=False)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False)

        return train_loader, val_loader, test_loader


def count_calls(func):
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        print(f"Function {func.__name__} has been called {wrapper.calls} times")
        # 将计数器的值作为参数传递给my_function
        return func(wrapper.calls, *args, **kwargs)
    wrapper.calls = 0
    return wrapper


@ count_calls
def Gesture_dataset_loader_A(call_count=0, batch_size=128, sequence_len=250, random_seed=42, LOSO=True):
    train_path = r"C:\Users\W\Desktop\导入\数据集\\Gesture-A\\training"
    test_path = r"C:\Users\W\Desktop\导入\数据集\\Gesture-A\\testing"

    "16名被试"

    class CustomDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            # 将数据返回为tensor
            data = torch.tensor(self.data[idx], dtype=torch.float32)
            # 将标签编码为数值
            label = self.labels[idx]
            return data, label

    def process_user_data(path, model="train"):
        """
        处理指定路径下的所有userData.mat文件，并提取accel、gyro、quaternion、emg数据。

        参数:
        path (str): 要处理的路径

        返回:
        None
        """
        accs, gyros, qua, emgs = [], [], [], []
        delete_id = []
        gesture_types = []

        # 获取路径下的所有子文件夹
        subfolders = [f.path for f in os.scandir(path) if f.is_dir()]

        for subfolder in subfolders:
            mat_file_path = os.path.join(subfolder, 'userData.mat')
            if os.path.exists(mat_file_path):
                try:
                    mat = scipy.io.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)
                    # print(f"Processing {mat_file_path}:")

                    user_data = mat['userData']
                    # variable = user_data.training if model == "train" else user_data.testing
                    variable = user_data.training

                    for i in range(len(variable)):

                        acc = np.array(variable[i].accel)
                        gyr = np.array(variable[i].gyro)
                        quaternion = np.array(variable[i].quaternions)

                        if len(acc) != 250:
                            # 如果长度小于250，则添加零直到达到250
                            if len(acc) < 250:
                                zeros_to_add = 250 - len(acc)
                                acc = np.concatenate((acc, np.zeros((zeros_to_add, 3))))
                                gyr = np.concatenate((gyr, np.zeros((zeros_to_add, 3))))
                                quaternion = np.concatenate((quaternion, np.zeros((zeros_to_add, 4))))
                            # 如果长度大于250，则移除超出的部分
                            elif len(acc) > 250:
                                acc = acc[:250]
                                gyr = gyr[:250]
                                quaternion = quaternion[:250]

                        accs.append(acc)
                        gyros.append(gyr)
                        qua.append(quaternion)

                except Exception as e:
                    ser_id_str = subfolder.split('user_')[1]
                    delete_id.append(int(ser_id_str))
                    # print(f"Failed to process {mat_file_path}: {e}")

        # print(delete_id)
        path_label = os.path.join(path, "Label")
        csv_files = glob(os.path.join(path_label, "*.csv"))
        csv_files.sort(key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))

        for file_path in csv_files:
            # 构造完整的文件路径
            # file_path = os.path.join(path_label, filename)
            label = pd.read_csv(file_path, header=None).to_numpy()
            gesture_types.append(label)

        total_label = np.array(gesture_types)
        total_label = np.delete(total_label, [id - 1 for id in delete_id], axis=0).squeeze().reshape(-1)

        total_data = np.concatenate((np.array(accs), np.array(gyros), np.array(qua)), axis=-1)
        return total_data, total_label

    def process_user_data_LOSO(path, model="train"):
        """
        处理指定路径下的所有userData.mat文件，并提取accel、gyro、quaternion、emg数据。
        """

        # 获取路径下的所有子文件夹

        useful_id = ["001", "024", "025", "026", "027", "032", "033", "035", "036", "037",
                     "038", "039", "040", "041", "042", "043"]
        # 获取有效路径，训练路径和测试路径
        test_id = useful_id[call_count-1]

        train_id = [uid for uid in useful_id if uid != test_id]

        train_data_path = [r"C:\Users\W\Desktop\导入\数据集\\Gesture-A\\training\\user_" + str(uid) for uid in train_id]
        test_data_path = [r"C:\Users\W\Desktop\导入\数据集\\Gesture-A\\testing\\user_" + test_id]

        train_id_int = [int(id) for id in train_id]
        test_id_int = [int(test_id)]
        print(f"选择被试id：{int(test_id)}")
        train_label_path = [r"C:\Users\W\Desktop\导入\数据集\\Gesture-A\\training\\Label\\gestureNames_"
                     + str(id)+".csv" for id in train_id_int]
        test_label_path = [r"C:\Users\W\Desktop\导入\数据集\\Gesture-A\\testing\\Label\\gestureNames_"
                     + str(id)+".csv" for id in test_id_int]

        # 训练数据部分
        train_accs, train_gyros, train_qua, train_emgs = [], [], [], []
        train_gesture_types = []

        for subfolder in train_data_path:
            mat_file_path = os.path.join(subfolder, 'userData.mat')

            mat = scipy.io.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)
            # print(f"Processing {mat_file_path}:")

            user_data = mat['userData']
            # variable = user_data.training if model == "train" else user_data.testing
            variable = user_data.training

            for i in range(len(variable)):

                acc = np.array(variable[i].accel)
                gyr = np.array(variable[i].gyro)
                quaternion = np.array(variable[i].quaternions)

                if len(acc) != 250:
                    # 如果长度小于250，则添加零直到达到250
                    if len(acc) < 250:
                        zeros_to_add = 250 - len(acc)
                        acc = np.concatenate((acc, np.zeros((zeros_to_add, 3))))
                        gyr = np.concatenate((gyr, np.zeros((zeros_to_add, 3))))
                        quaternion = np.concatenate((quaternion, np.zeros((zeros_to_add, 4))))
                    # 如果长度大于250，则移除超出的部分
                    elif len(acc) > 250:
                        acc = acc[:250]
                        gyr = gyr[:250]
                        quaternion = quaternion[:250]

                train_accs.append(acc)
                train_gyros.append(gyr)
                train_qua.append(quaternion)

        train_data = np.concatenate((np.array(train_accs), np.array(train_gyros), np.array(train_qua)), axis=-1)

        # 训练集label
        for file_path in train_label_path:
            # 构造完整的文件路径
            label = pd.read_csv(file_path, header=None).to_numpy()
            train_gesture_types.append(label)
        train_label = np.array(train_gesture_types).squeeze().reshape(-1)

        # 测试数据
        test_accs, test_gyros, test_qua, test_emgs = [], [], [], []
        test_gesture_types = []

        for subfolder in test_data_path:
            mat_file_path = os.path.join(subfolder, 'userData.mat')

            mat = scipy.io.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)
            # print(f"Processing {mat_file_path}:")

            user_data = mat['userData']
            # variable = user_data.training if model == "train" else user_data.testing
            variable = user_data.training

            for i in range(len(variable)):

                acc = np.array(variable[i].accel)
                gyr = np.array(variable[i].gyro)
                quaternion = np.array(variable[i].quaternions)

                if len(acc) != 250:
                    # 如果长度小于250，则添加零直到达到250
                    if len(acc) < 250:
                        zeros_to_add = 250 - len(acc)
                        acc = np.concatenate((acc, np.zeros((zeros_to_add, 3))))
                        gyr = np.concatenate((gyr, np.zeros((zeros_to_add, 3))))
                        quaternion = np.concatenate((quaternion, np.zeros((zeros_to_add, 4))))
                    # 如果长度大于250，则移除超出的部分
                    elif len(acc) > 250:
                        acc = acc[:250]
                        gyr = gyr[:250]
                        quaternion = quaternion[:250]

                test_accs.append(acc)
                test_gyros.append(gyr)
                test_qua.append(quaternion)
        test_data = np.concatenate((np.array(test_accs), np.array(test_gyros), np.array(test_qua)), axis=-1)

        for file_path in test_label_path:
            # 构造完整的文件路径
            label = pd.read_csv(file_path, header=None).to_numpy()
            test_gesture_types.append(label)
        test_label = np.array(test_gesture_types).squeeze().reshape(-1)

        return train_data, train_label,test_data,test_label

    if LOSO:
        trian_data, train_label, test_data, test_label = process_user_data_LOSO(train_path)
    else:
        trian_data, train_label = process_user_data(train_path)
    # test_data, test_label = process_user_data(test_path)

    # 整数编码
    labels = np.unique(train_label)
    label_to_index = {label: index for index, label in enumerate(labels)}

    if LOSO:
        train_labels_encoded = np.array([label_to_index[label] for label in train_label])
        test_labels_encoded = np.array([label_to_index[label] for label in test_label])
        train_dataset = CustomDataset(trian_data, train_labels_encoded)
        test_dataset = CustomDataset(test_data, test_labels_encoded)

        X_train = torch.from_numpy(train_dataset.data).to(torch.float32)
        y_train = torch.from_numpy(train_dataset.labels).to(torch.int64)

        X_test = torch.from_numpy(test_dataset.data).to(torch.float32)
        y_test = torch.from_numpy(test_dataset.labels).to(torch.int64)

        train = torch.utils.data.TensorDataset(X_train, y_train)
        test = torch.utils.data.TensorDataset(X_test, y_test)

        # data loader
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, drop_last=False)

        return train_loader,test_loader,test_loader

    else:
        # 将字符串标签转换为整数
        labels_encoded = np.array([label_to_index[label] for label in train_label])

        # 创建Dataset实例
        dataset = CustomDataset(trian_data, labels_encoded)

        """
            test_dataset = CustomDataset(test_data, test_label, label_encoder)
    
            # 创建DataLoader实例
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        """

        X_train, X_temp, y_train, y_temp = train_test_split(
            dataset.data,
            dataset.labels.ravel(),
            test_size=0.2,
            random_state=random_seed
        )

        # 然后从训练集中分割出验证集
        X_test, X_val, y_test, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            random_state=random_seed
        )

        X_train = torch.from_numpy(X_train).to(torch.float32)
        y_train = torch.from_numpy(y_train).to(torch.int64)
        X_val = torch.from_numpy(X_val).to(torch.float32)
        y_val = torch.from_numpy(y_val).to(torch.int64)  # 确保数据类型正确
        X_test = torch.from_numpy(X_test).to(torch.float32)
        y_test = torch.from_numpy(y_test).to(torch.int64)

        train = torch.utils.data.TensorDataset(X_train, y_train)
        val = torch.utils.data.TensorDataset(X_val, y_val)
        test = torch.utils.data.TensorDataset(X_test, y_test)

        # data loader
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=False)  # 验证集通常不打乱数据
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, drop_last=False)

        return train_loader, val_loader, test_loader


@ count_calls
def Gesture_dataset_loader_B(call_count=0, batch_size=128, sequence_len=20, LOSO=False):
    """
    Gesture-B-recognize using IMU module

    40000+ 6-dimension samples from 6 gestures
    实验室自制数据集,TIM2024
    return(batch_size,sequence_len,6)
    """

    def rawarray2dataloader(x: np.array, y: np.array, LOSO_test=False):

        # 默认情况/LOSO_train：0.8,0.5
        # LOSO_test情况：0.01,0.99

        X_train, X_temp, y_train, y_temp = train_test_split(x, y, train_size=0.01 if LOSO_test else 0.8, random_state=42,
                                                            shuffle=True)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=0.01 if LOSO_test else 0.5, random_state=42, shuffle=True)

        X_train = torch.from_numpy(X_train).to(torch.float32)
        y_train = torch.from_numpy(y_train).type(torch.LongTensor)  # data eval_type is long

        X_val = torch.from_numpy(X_val).to(torch.float32)
        y_val = torch.from_numpy(y_val).type(torch.LongTensor)  # data eval_type is

        # create feature and targets tensor for test set.
        X_test = torch.from_numpy(X_test).to(torch.float32)
        y_test = torch.from_numpy(y_test).type(torch.LongTensor)  # data eval_type is long

        # Pytorch train and test sets
        train = torch.utils.data.TensorDataset(X_train, y_train)
        val = torch.utils.data.TensorDataset(X_val, y_val)
        test = torch.utils.data.TensorDataset(X_test, y_test)

        # data loader
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=False)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False)

        return train_loader, val_loader, test_loader

    "7个被试"
    # if LOSO:
    #     total_path = r"/开源数据集/实验室数据集-手势识别/LOSO-train"
    #     data_files = [f for f in os.listdir(total_path) if f.startswith("dataSquence") and f.endswith(".csv")]
    #     label_files = [f for f in os.listdir(total_path) if f.startswith("labels") and f.endswith(".csv")]
    #
    #     test_datas = []
    #     train_datas = []
    #     test_labels = []
    #     train_labels= []
    #
    #     for file in data_files:
    #         file_name_without_extension_id = int(os.path.splitext(file)[0][-1])-1
    #         # 测试集
    #         if file_name_without_extension_id == call_count:
    #             file_path = os.path.join(total_path, file)
    #             data = pd.read_csv(file_path, dtype=np.float32)
    #
    #             x = data.iloc[:, 1:].values
    #             x_unsquueze = []
    #             for i in range(len(x)):
    #                 x_unsquueze.append(x[i].reshape(6, sequence_len).T)
    #             test_datas.append(np.array(x_unsquueze))
    #
    #         # 训练集
    #         else:
    #             file_path = os.path.join(total_path, file)
    #             data = pd.read_csv(file_path, dtype=np.float32)
    #
    #             x = data.iloc[:, 1:].values
    #             x_unsquueze = []
    #             for i in range(len(x)):
    #                 x_unsquueze.append(x[i].reshape(6, sequence_len).T)
    #             train_datas.append(np.array(x_unsquueze))
    #
    #     for file in label_files:
    #         file_name_without_extension_id = int(os.path.splitext(file)[0][-1])-1
    #         # 测试集
    #         if file_name_without_extension_id == call_count:
    #             file_path = os.path.join(total_path, file)
    #             labels = pd.read_csv(file_path, dtype=np.float32)
    #             test_labels.append(labels.iloc[:, 1].values)
    #         # 训练集
    #         else:
    #             file_path = os.path.join(total_path, file)
    #             labels = pd.read_csv(file_path, dtype=np.float32)
    #             train_labels.append(labels.iloc[:, 1].values)
    #
    #     # list2array
    #     test_data = np.array(test_datas[0])
    #     test_label = np.array(test_labels[0])
    #
    #     train_data = train_datas[0]
    #     for data in train_datas:
    #         train_data = np.concatenate((train_data, data), axis=0)
    #
    #     train_label = train_labels[0]
    #     for label in train_labels:
    #         train_label = np.concatenate((train_label, label), axis=0)
    #
    #     # array2datasetloader
    #     X_train = torch.from_numpy(train_data).to(torch.float32)
    #     y_train = torch.from_numpy(train_label).eval_type(torch.LongTensor)  # data eval_type is long
    #
    #     # create feature and targets tensor for test set.
    #     X_test = torch.from_numpy(test_data).to(torch.float32)
    #     y_test = torch.from_numpy(test_label).eval_type(torch.LongTensor)  # data eval_type is long
    #
    #     # Pytorch train and test sets
    #     train = torch.utils.data.TensorDataset(X_train, y_train)
    #     test = torch.utils.data.TensorDataset(X_test, y_test)
    #
    #     # data loader
    #     train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=False)
    #     test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, drop_last=False)
    #
    #     return train_loader, test_loader, test_loader

    if LOSO:
        train_path = r"C:\Users\admin\Desktop\emergency gesture re\开源数据集\实验室数据集-手势识别\LOSO-train"
        test_path = r"C:\Users\admin\Desktop\emergency gesture re\开源数据集\实验室数据集-手势识别\LOSO-test"
        train_data_files = [f for f in os.listdir(train_path) if f.startswith("dataSquence") and f.endswith(".csv")]
        train_label_files = [f for f in os.listdir(train_path) if f.startswith("labels") and f.endswith(".csv")]
        test_data_files = [f for f in os.listdir(test_path) if f.startswith("dataSquence") and f.endswith(".csv")]
        test_label_files = [f for f in os.listdir(test_path) if f.startswith("labels") and f.endswith(".csv")]

        if call_count == 1:
            data_mean = [0.286650747, -0.372543332, 0.3402251, 1.2493385, -2.4693056, -0.3384287]
            data_std = [0.905598, 0.575761, 0.89109, 66.758979, 266.737187, 92.8107196]
        if call_count == 2:
            data_mean = [0.25779139, -0.3760106, 0.3619041, 1.5451783, -2.548304, -0.3704956]
            data_std = [0.88237536, 0.568916, 0.8623351, 64.85947, 259.74579, 87.641999]
        elif call_count == 3:
            data_mean = [0.280378777, -0.376986, 0.3488899, 1.2977789, -2.791987, -0.73653036]
            data_std = [0.8895487, 0.577911, 0.8863859, 66.89028, 262.3927125, 92.611627]
        elif call_count == 4:
            data_mean = [0.274561579, -0.39420622, 0.3428919, 1.14678149, -2.378757, -0.878703]
            data_std = [0.87879, 0.562816, 0.87856, 63.874321, 258.244575, 91.448759]
        elif call_count == 5:
            data_mean = [0.30856281, -0.385359, 0.3247456, 1.2890146, -2.6775305, -0.56017]
            data_std = [0.9173626, 0.57928, 0.9164489, 68.01585402, 272.4483642, 94.167666]
        elif call_count == 6:
            data_mean = [0.3038817, -0.385617, 0.33381454, 0.7601886, -2.8088241, -0.4801621]
            data_std = [0.917697, 0.578926, 0.9201366, 67.199596, 271.5046503, 94.15756]
        elif call_count == 7:
            data_mean = [0.28107392, -0.3801060, 0.34101424, 1.21890021, -2.5395044, -0.251780158]
            data_std = [0.89164907, 0.57648136, 0.89019331, 65.7201829, 258.948009, 90.14901052]

        test_datas = []
        train_datas = []
        test_labels = []
        train_labels= []

        for file in train_data_files:
            file_name_without_extension_id = int(os.path.splitext(file)[0][-1])
            if file_name_without_extension_id == call_count:
                file_path = os.path.join(train_path, file)
                data = pd.read_csv(file_path, dtype=np.float32)
                x = data.iloc[:, 1:].values
                x_unsquueze = []
                for i in range(len(x)):
                    x_unsquueze.append(x[i].reshape(6, sequence_len).T)
                train_datas.append(np.array(x_unsquueze))

        for file in train_label_files:
            file_name_without_extension_id = int(os.path.splitext(file)[0][-1])
            if file_name_without_extension_id == call_count:
                file_path = os.path.join(train_path, file)
                labels = pd.read_csv(file_path, dtype=np.float32)
                train_labels.append(labels.iloc[:, 1].values)

        for file in test_data_files:
            file_name_without_extension_id = int(os.path.splitext(file)[0][-1])
            if file_name_without_extension_id == call_count:
                file_path = os.path.join(test_path, file)
                data = pd.read_csv(file_path, dtype=np.float32)
                x = data.iloc[:, 1:].values
                x_unsquueze = []
                for i in range(len(x)):
                    x_test = x[i].reshape(6, sequence_len).T
                    for j in range(6):
                        x_test[:, j] = (x_test[:, j] - data_mean[j]) / data_std[j]
                    x_unsquueze.append(x_test)
                test_datas.append(np.array(x_unsquueze))

        for file in test_label_files:
            file_name_without_extension_id = int(os.path.splitext(file)[0][-1])
            if file_name_without_extension_id == call_count:
                file_path = os.path.join(test_path, file)
                labels = pd.read_csv(file_path, dtype=np.float32)
                test_labels.append(labels.iloc[:, 1].values)

        # array2datasetloader
        X_train = np.concatenate(train_datas, axis=0)
        y_train = np.concatenate(train_labels, axis=0)
        X_test = np.concatenate(test_datas, axis=0)
        y_test = np.concatenate(test_labels, axis=0)

        train_loader,_, _ = rawarray2dataloader(X_train, y_train, LOSO_test=False)
        _, val_loader, test_loader = rawarray2dataloader(X_test, y_test, LOSO_test=True)

        return train_loader, test_loader, test_loader

    else:
        if sequence_len == 20:
            data = pd.read_csv(
                r"C:\Users\admin\Desktop\emergency gesture re\开源数据集\实验室数据集-手势识别\dataSquence20.csv",
                dtype=np.float32)
            # data = pd.read_csv(
            #     r"C:\Users\admin\Desktop\emergency gesture re\开源数据集\实验室数据集-手势识别\LOSO-train\dataSquence20_1.csv",
            #     dtype=np.float32)

            labels = pd.read_csv(
                r"C:\Users\admin\Desktop\emergency gesture re\开源数据集\实验室数据集-手势识别\labels20.csv",
                dtype=np.float32)
            # labels = pd.read_csv(
            #     r"C:\Users\admin\Desktop\emergency gesture re\开源数据集\实验室数据集-手势识别\LOSO-train\labels20_1.csv",
            #     dtype=np.float32)

        else:
            print("数据长度错误")

        x = data.iloc[:, 1:].values
        x_unsquueze = []
        for i in range(len(x)):
            x_unsquueze.append(x[i].reshape(6, sequence_len).T)
        x = np.array(x_unsquueze)
        y = labels.iloc[:, 1].values

        train_loader, val_loader, test_loader = rawarray2dataloader(x, y)
        return train_loader, val_loader, test_loader


@ count_calls
def AirWrite_dataset_loader(test_numbers=0, batch_size=128, sequence_len=155, id=1, LOSO=False, random_seed=42):
    """
    AiRWrite-recognize using IMU module

    6-dimension samples from 26 gestures

    ID1：55 subjects，62HZ
    T. Yanay and E. Shmueli, “Air-writing recognition using smart-bands,” Pervasive and Mobile Computing, vol. 66, p. 101183, 2020.

    ID2：20 subjects，62HZ
    SCLAiR : Supervised Contrastive Learning for User and Device Independent Airwriting Recognition

    return(batch_size,sequence_len,6)
    """
    if id == 1:
        path = r"C:\Users\W\Desktop\导入\数据集\Subject_wise_data3"
    elif id == 2:
        path = r"C:\Users\W\Desktop\导入\数据集\Recorded_preprocessed_data"

    if LOSO:
        numbers = [int(re.search(r'\d+', filename).group()) for filename in os.listdir(path) if
                   filename.startswith('X')]
        max_number = max(numbers) if numbers else None

        # 修改这个，当AW-A时从0到54，AW-B时从1到20
        # random_number = np.random.randint(0, max_number - 1)

        random_number = test_numbers

        dataset_train = MyDataset(path, sequence_len, test_num=random_number, type="train")
        dataset_test = MyDataset(path, sequence_len, test_num=random_number, type="test")

        dataset_tensor = torch.tensor(dataset_train.data, dtype=torch.float32)
        mean = dataset_tensor.mean(dim=(0, 2))
        std = dataset_tensor.std(dim=(0, 2))
        dataset_train.data = ((dataset_tensor - mean.view(1, -1, 1)) / std.view(1, -1, 1)).numpy()

        dataset_tensor = torch.tensor(dataset_test.data, dtype=torch.float32)
        mean = dataset_tensor.mean(dim=(0, 2))
        std = dataset_tensor.std(dim=(0, 2))
        dataset_test.data = ((dataset_tensor - mean.view(1, -1, 1)) / std.view(1, -1, 1)).numpy()

        X_train = torch.from_numpy(dataset_train.data).to(torch.float32)
        y_train = torch.from_numpy(dataset_train.labels.ravel()).int()
        X_test = torch.from_numpy(dataset_test.data).to(torch.float32)
        y_test = torch.from_numpy(dataset_test.labels.ravel()).int()

        train = torch.utils.data.TensorDataset(X_train, y_train)
        test = torch.utils.data.TensorDataset(X_test, y_test)

        # data loader
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True,drop_last=True)

        return train_loader, test_loader, test_loader

    else:
        dataset = MyDataset(path, sequence_len)
        dataset_tensor = torch.tensor(dataset.data, dtype=torch.float32)
        mean = dataset_tensor.mean(dim=(0, 2))
        std = dataset_tensor.std(dim=(0, 2))
        dataset.data = ((dataset_tensor - mean.view(1, -1, 1)) / std.view(1, -1, 1)).numpy()
        X_train, X_temp, y_train, y_temp = train_test_split(
            dataset.data,
            dataset.labels.ravel(),
            test_size=0.2,
            random_state=random_seed
        )

        # 然后从训练集中分割出验证集
        X_test, X_val, y_test, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            random_state=random_seed
        )

        X_train = torch.from_numpy(X_train).to(torch.float32)
        y_train = torch.from_numpy(y_train).to(torch.int64)
        X_val = torch.from_numpy(X_val).to(torch.float32)
        y_val = torch.from_numpy(y_val).to(torch.int64)  # 确保数据类型正确
        X_test = torch.from_numpy(X_test).to(torch.float32)
        y_test = torch.from_numpy(y_test).to(torch.int64)

        train = torch.utils.data.TensorDataset(X_train, y_train)
        val = torch.utils.data.TensorDataset(X_val, y_val)
        test = torch.utils.data.TensorDataset(X_test, y_test)

        # data loader
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=False)  # 验证集通常不打乱数据
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, drop_last=False)

        return train_loader, val_loader, test_loader


def EMG_dataset_loader(batch_size=128, sequence_len=40, LOSO=False):
    """
    Gesture-B-recognize using EMG module

    37908 16-dimension samples from 52 gestures
    E1：12,E2:17，E3：23，E4：9

    Ninapro DB5 dataset
    Pizzolato et al., Comparison of six electromyography acquisition setups on hand movement classification tasks, PLOS One, 2017

    return(batch_size,sequence_len,16)
    """
    if LOSO:
        subprocess.run(["python", r"C:\Users\admin\Desktop\emergency gesture re\EMG数据预处理\ninaweb_sEMG_envelop_divide_by_subject.py",
                        str(sequence_len)])

    else:
        subprocess.run(["python", r"C:\Users\admin\Desktop\emergency gesture re\EMG数据预处理\ninaweb_sEMG_envelop.py",
                        str(sequence_len)])

    train_data_path = r"C:\Users\admin\Desktop\emergency gesture re\开源数据集\EMG\EMG_train.npy"
    test_data_path = r"C:\Users\admin\Desktop\emergency gesture re\开源数据集\EMG\EMG_test.npy"
    train_labels_path = r'C:\Users\admin\Desktop\emergency gesture re\开源数据集\EMG\label_train.npy'
    test_labels_path = r'C:\Users\admin\Desktop\emergency gesture re\开源数据集\EMG\label_test.npy'

    # 创建数据集
    train_dataset = NumpyDataset(data_file_path=train_data_path, labels_file_path=train_labels_path , transform=None)
    test_dataset = NumpyDataset(data_file_path=test_data_path, labels_file_path=test_labels_path, transform=None)

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # min_label_train = np.min(train_dataset.labels)
    # max_label_train = np.max(train_dataset.labels)
    # print("Min label:", min_label_train)
    # print("Max label:", max_label_train)
    # min_label_test = np.min(test_dataset.labels)
    # max_label_test = np.max(test_dataset.labels)
    # print("Min label:", min_label_test)
    # print("Max label:", max_label_test)
    unique_labels = np.unique(test_dataset.labels)
    unique_labels = np.unique(train_dataset.labels)

    return train_dataloader, test_dataloader


def load_data(args):
    """
    根据给定的参数加载数据集，并返回数据集的序列长度、类别数量、通道数量，以及训练、验证和测试数据的加载器。

    参数:
    - args: 包含数据集名称、批次大小和是否使用留一法交叉验证的参数对象

    返回:
    - seq_len: 数据集的序列长度
    - num_classes: 数据集的类别数量
    - num_channel: 数据集的通道数量
    - train_loader: 训练数据的加载器
    - val_loader: 验证数据的加载器
    - test_loader: 测试数据的加载器
    """
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    useLeaveOneOutCrossValidation = args.useLeaveOneOutCrossValidation

    # 定义数据集参数字典
    dataset_dict = {
        "AW-A": {"seq_len": 155, "batch_size": batch_size, "num_classes": 26,
                 "leaveOneout": useLeaveOneOutCrossValidation, "num_channel": 6},
        "AW-B": {"seq_len": 155, "batch_size": batch_size, "num_classes": 26,
                 "leaveOneout": useLeaveOneOutCrossValidation, "num_channel": 6},
        "Gesture-A": {"seq_len": 250, "batch_size": batch_size, "num_classes": 12,
                      "num_channel": 10},
        "Gesture-B": {"seq_len": 20, "batch_size": batch_size, "num_classes": 6,
                      "num_channel": 6},
        "HAR-A": {"seq_len": 128, "batch_size": batch_size, "num_classes": 6,
                  "num_channel": 9},
        "HAR-B": {"seq_len": 151, "batch_size": batch_size, "num_classes": 17,
                  "num_channel": 3},
        "HAR-C": {"seq_len": 256, "batch_size": batch_size, "num_classes": 6,
                  "num_channel": 9},
    }

    # 从字典中获取数据集参数
    seq_len = dataset_dict[dataset_name]["seq_len"]
    num_classes = dataset_dict[dataset_name]["num_classes"]
    num_channel = dataset_dict[dataset_name]["num_channel"]

    # 定义数据加载器字典
    dataset_loaders = {
        "AW-A": lambda: AirWrite_dataset_loader(batch_size=batch_size, sequence_len=seq_len, id=1, LOSO=useLeaveOneOutCrossValidation),
        "AW-B": lambda: AirWrite_dataset_loader(batch_size=batch_size, sequence_len=seq_len, id=2, LOSO=useLeaveOneOutCrossValidation),
        "Gesture-A": lambda: Gesture_dataset_loader_A(batch_size=batch_size, sequence_len=seq_len),
        "Gesture-B": lambda: Gesture_dataset_loader_B(batch_size=batch_size, sequence_len=seq_len, LOSO=useLeaveOneOutCrossValidation),
        "HAR-A": lambda: HAR_dataset_loader(batch_size=batch_size, sequence_len=seq_len, id=1),
        "HAR-B": lambda: HAR_dataset_loader(batch_size=batch_size, sequence_len=seq_len, id=2),
        "HAR-C": lambda: HAR_dataset_loader(batch_size=batch_size, sequence_len=seq_len, id=3),
    }

    try:
        # 根据数据集名称获取数据加载器
        train_loader, val_loader, test_loader = dataset_loaders[dataset_name]()
    except KeyError:
        raise Warning("不支持的数据集！")

    return seq_len, num_classes, num_channel, train_loader, val_loader, test_loader


def process_dataset_text(args):
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    seq_len, num_classes, num_channel, train_loader, val_loader, test_loader = load_data(args)

    data_val = val_loader.dataset
    data_train = train_loader.dataset
    data_test = test_loader.dataset

    if dataset_name.startswith("AW"):
        prefix = "For the dataset obtained from the wearable IMU-based aerial gesture recognition process, please select one of the 26 labels corresponding to the alphabet letters from A to Z, based on the provided information regarding the detected gestures.\nAt this moment, the individual is performing a particular aerial gesture of "
        midfix = "In the scenario of aerial gesture recognition using signals collected by wearable IMUs, 26 labels are utilized to represent the various aerial gestures corresponding to the alphabet letters from A to Z.\n"

        # prefix = "For the AirWrite dataset, please select one of the 26 labels corresponding to the alphabet letters from A to Z, using the provided information.\nAt this moment, the individual is in a specific state of "
        # midfix = "In the clinical assessment of AirWrite data, 26 labels are used to represent the alphabet letters from A to Z.\n"

        signals_text_train = []
        signals_text_val = []
        signals_text_test = []

        for sample in data_train:
            data, label = sample
            if torch.isnan(data).any():
                continue
            text = chr(label.item() + 65)  # 将label转换为对应的字母
            if text == '':
                continue

            text = midfix + prefix + text
            label_vector = label

            if data.shape == (seq_len, num_channel):
                signals_text_train.append((text, data, label_vector))

        for sample in data_val:
            data, label = sample
            if torch.isnan(data).any():
                continue
            text = chr(label.item() + 65)  # 将label转换为对应的字母
            if text == '':
                continue

            text = midfix + prefix + text
            label_vector = label

            if data.shape == (seq_len, num_channel):
                signals_text_val.append((text, data, label_vector))

        for sample in data_test:
            data, label = sample
            if torch.isnan(data).any():
                continue
            text = chr(label.item() + 65)  # 将label转换为对应的字母
            if text == '':
                continue

            text = midfix + prefix + text
            label_vector = label

            if data.shape == (seq_len, num_channel):
                signals_text_test.append((text, data, label_vector))
        
        return signals_text_train, signals_text_val, signals_text_test

    elif dataset_name.startswith("Gesture"):
        if dataset_name == "Gesture-A":
            prefix = "For the Gesture-A dataset, please choose one of the 12 labels and analyze the individual's condition for a possible epilepsy diagnosis based on the provided information.\nAt this moment, the individual is existing in a particular state of "
            midfix = "In the clinical evaluation of Gesture-A data, 12 labels are used to denote the patient's state: 'no abnormalities' for normal conditions and 'epileptic seizure' for seizure activity.\n"

            for data, label in zip(data_train[0],data_train[1]):
                if torch.isnan(data).any():
                    continue

                if label.item() == 0:
                    text = 'no abnormalities'
                elif label.item() == 1:
                    text = 'epileptic seizure'
                else:
                    text = ''

                if text == '':
                    continue

        elif dataset_name == "Gesture-B":
            prefix = "For the Gesture-B dataset, please choose one of the 6 labels and analyze the individual's condition for a possible epilepsy diagnosis based on the provided information.\nAt this moment, the individual is existing in a particular state of "
            midfix = "In the clinical evaluation of Gesture-B data, 6 labels are used to denote the patient's state: 'no abnormalities' for normal conditions and 'epileptic seizure' for seizure activity.\n"

            for data, label in zip(data_train[0],data_train[1]):
                if torch.isnan(data).any():
                    continue

                if label.item() == 0:
                    text = 'no abnormalities'
                elif label.item() == 1:
                    text = 'epileptic seizure'
                else:
                    text = ''

                if text == '':
                    continue

    elif dataset_name.startswith("HAR"):
        if dataset_name == "HAR-A":
            prefix = "Please choose one activity from the previously mentioned six options and analyze the individual's physical activity based on the provided information.\nThe individual is currently engaged in "
            midfix = 'Physical activities such as walking, ascending stairs, descending stairs, sitting, standing, and lying down are recorded using mobile phone sensors.\n'

            signals_text_train = []
            signals_text_val = []
            signals_text_test = []

            activity_labels = {
                0: 'walking',
                1: 'ascending stairs',
                2: 'descending stairs',
                3: 'sitting',
                4: 'standing',
                5: 'lying down'
            }

            for sample in data_train:
                data, label = sample

                if torch.isnan(data).any():
                    continue

                text = activity_labels.get(int(label.item()), '')
                if not text:
                    continue

                text = midfix + prefix + text
                label_vector = label

                if data.shape == (seq_len, num_channel):
                    signals_text_train.append((text, data, label_vector))

            for sample in data_val:
                data, label = sample
                if torch.isnan(data).any():
                    continue

                text = activity_labels.get(int(label.item()), '')
                if not text:
                    continue

                text = midfix + prefix + text
                label_vector = label

                if data.shape == (seq_len, num_channel):
                    signals_text_val.append((text, data, label_vector))

            for sample in data_test:
                data, label = sample

                if torch.isnan(data).any():
                    continue

                text = activity_labels.get(int(label.item()), '')
                if not text:
                    continue

                text = midfix + prefix + text
                label_vector = label

                if data.shape == (seq_len, num_channel):
                    signals_text_test.append((text, data, label_vector))

                return signals_text_train, signals_text_val, signals_text_val

        elif dataset_name == "HAR-B":
            prefix = "For the HAR-B dataset, please select one of the 17 labels and evaluate the individual's condition for potential epilepsy diagnosis based on the provided information.\nAt this moment, the individual is in a specific state of "
            midfix = "In the clinical assessment of HAR-B data, 17 labels are used to indicate the patient's state: 'no abnormalities' for normal conditions and 'epileptic seizure' for seizure activity.\n"

            for data, label in zip(data_train[0],data_train[1]):
                if torch.isnan(data).any():
                    continue

                if label.item() == 0:
                    text = 'no abnormalities'
                elif label.item() == 1:
                    text = 'epileptic seizure'
                else:
                    text = ''

                if text == '':
                    continue

        elif dataset_name == "HAR-C":
            prefix = "Please choose one activity from the previously mentioned six options and analyze the individual's physical activity based on the provided information.\nThe individual is currently engaged in "
            midfix = 'Physical activities such as walking, ascending stairs, descending stairs, sitting, standing, and lying down are recorded using mobile phone sensors.\n'

            signals_text_train = []
            signals_text_val = []
            signals_text_test = []

            for data, label in zip(data_train[0], data_train[1]):
                if torch.isnan(data).any():
                    continue

                if int(label) == 0:
                    text = 'walking'
                elif int(label) == 1:
                    text = 'ascending stairs'
                elif int(label) == 2:
                    text = 'descending stairs'
                elif int(label) == 3:
                    text = 'sitting'
                elif int(label) == 4:
                    text = 'standing'
                elif int(label) == 5:
                    text = 'lying down'
                else:
                    text = ''

                if text == '':
                    continue

                text = midfix + prefix + text
                label_vector = label

                if data.shape == (seq_len, num_channel):
                    signals_text_train.append((text, data, label_vector))

                return signals_text_train, signals_text_val, signals_text_test
    else:
        raise ValueError("Unsupported dataset name.")



    # np.random.shuffle(normalized_samples)
    # split = int(0.8 * len(normalized_samples))
    # samples_train = normalized_samples[:split]
    # samples_test = normalized_samples[split:]

    # with open('samples_train.pkl', 'wb') as file:
    #     pickle.dump(normalized_samples_train, file)
    # with open('samples_test.pkl', 'wb') as file:
    #     pickle.dump(normalized_samples_test, file)


class TestDataGenerator:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.batch_size = 32
        self.useLeaveOneOutCrossValidation = False

    def generate_test(self):
        if self.dataset_name.startswith("Gesture"):
            print("生成Gesture数据集的测试")
            process_dataset_text(self)
        elif self.dataset_name.startswith("HAR"):
            print("生成HAR数据集的测试")
            process_dataset_text(self)
        elif self.dataset_name.startswith("AW"):
            print("生成AW数据集的测试")
            process_dataset_text(self)
        else:
            raise ValueError("Unsupported dataset name.")
