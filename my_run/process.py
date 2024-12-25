import os
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import f1_score
from tqdm import tqdm
import time
import datetime
import torch.utils.tensorboard as tensorboard
from ray import train, tune
from ray.tune import Callback

import wandb

class LossCalculator:
    def __init__(self, model, latent_loss_weight=0.25):
        self.model = model
        self.latent_loss_weight = latent_loss_weight
        self.mse = nn.MSELoss()

    def compute(self, batch):
        """
        计算给定批次数据的交叉熵损失。

        参数:
        - batch: 一个包含输入序列和对应标签的批次数据，期望是可解包的格式（如元组、列表等），
                其中第一个元素是输入序列，第二个元素是标签。

        返回:
        - loss: 计算得到的交叉熵损失值，如果出现异常情况则返回None。
        """
        # 检查batch是否可解包为期望的两个元素（输入序列和标签）
        if not isinstance(batch, (tuple, list)) or len(batch)!= 2:
            print("Error: batch should be a tuple or list with exactly two elements.")
            return None

        seqs, labels = batch
        # 检查输入序列seqs是否是张量类型，若不是则尝试转换
        if not isinstance(seqs, torch.Tensor):
            try:
                seqs = torch.tensor(seqs)
            except:
                print("Error: Failed to convert input sequences to tensor.")
                return None

        # 检查标签labels是否是张量类型，若不是则尝试转换
        if not isinstance(labels, torch.Tensor):
            try:
                labels = torch.tensor(labels)
            except:
                print("Error: Failed to convert labels to tensor.")
                return None

        # 通过模型获取输出
        out = self.model(seqs, None, None, None)
        # 检查模型输出out是否是张量类型
        if not isinstance(out, torch.Tensor):
            print("Error: Model output is not a tensor.")
            return None

        # 检查模型输出和标签的维度情况，确保它们符合交叉熵损失计算的要求
        if out.dim() < 2:
            print("Error: Model output should have at least two dimensions for cross-entropy loss calculation.")
            return None
        if labels.dim() == 0:
            print("Error: Labels should have at least one dimension.")
            return None

        # 获取输出的批量大小（假设输出维度格式为（批量大小，类别数等其他维度））
        out_batch_size = out.shape[0]
        # 获取标签的批量大小
        labels_batch_size = labels.shape[0]
        # 检查批量大小是否匹配
        if out_batch_size!= labels_batch_size:
            print(f"Error: Input batch_size ({out_batch_size}) does not match target batch_size ({labels_batch_size}).")
            return None

        try:
            # 计算交叉熵损失
            loss = nn.CrossEntropyLoss()(out, labels.long().squeeze())
        except Exception as e:
            print(f"Error occurred while calculating cross-entropy loss: {str(e)}")
            return None

        return loss
    
    def compute2(self, batch):
        seqs = batch
        out, latent_loss, _ = self.model(seqs)
        recon_loss = self.mse(out, seqs)
        latent_loss = latent_loss.mean()
        loss = recon_loss + self.latent_loss_weight * latent_loss
        return loss


class Trainer():
    def __init__(self, args, model, train_loader, test_loader, verbose=False):
        self.args = args
        self.verbose = verbose
        self.device = args.device
        print(f"使用设备: {self.device}")
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_decay = args.lr_decay_rate
        self.lr_decay_steps = args.lr_decay_steps
        self.weight_decay = args.weight_decay

        self.cr = LossCalculator(self.model)

        self.num_epoch = args.num_epoch
        self.eval_per_steps = args.eval_per_steps
        self.save_path = args.save_path

        self.step = 0
        self.best_metric = [-1e9,-1e9] #acc,F1
        self.metric = 'acc' #用于评估的指标
        log_dir = os.path.join("./logs",self.args.dataset_name)
        log_dir = os.path.join(log_dir,datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        print(log_dir)

    def train(self):
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.weight_decay)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_decay ** step, verbose=self.verbose)
        
        pbar = tqdm(range(self.num_epoch), 
                   desc="训练进度",
                   disable=False,
                   ncols=self.num_epoch)

        for epoch in pbar:
            loss_epoch, time_cost = self._train_one_epoch(epoch)
            self.result_file = open(self.save_path + '/result.txt', 'a+')
            log_msg = f'Epoch [{epoch+1}/{self.num_epoch}] Loss: {loss_epoch:.4f} Time: {time_cost:.2f}s'
            print(log_msg)
            print(log_msg, file=self.result_file)
            self.result_file.close()
            
        print(f"训练完成! 最佳指标: acc:{self.best_metric[0]:.4f} F1:{self.best_metric[1]:.4f}")
        return self.best_metric, loss_epoch
    
    def _train_one_epoch(self, epoch):
        t0 = time.perf_counter()
        self.model.train()
        
        pbar = tqdm(self.train_loader, 
                   desc="训练进度",
                   disable=not self.verbose,
                   ncols=self.num_epoch)

        loss_sum = 0
        total_correct = 0
        total_f1 = 0
        num_samples_processed = 0  # 用于记录已处理的样本总数

        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            data = [tensor.to(self.device) for tensor in batch]
            loss = self.cr.compute(data)
            loss_sum += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            # 计算训练过程中的准确率和F1值
            output = self.model(data[0], None, None, None)
            pred = output.argmax(dim=1, keepdim=True)
            target = data[1].view(pred.shape)
            correct = pred.eq(target).sum().item()
            total_correct += correct
            num_samples_processed += pred.numel()

            pred_labels = pred.cpu().numpy().flatten()
            target_labels = target.cpu().numpy().flatten()
            try:
                f1 = f1_score(target_labels, pred_labels, average='macro')
                total_f1 += f1
            except:
                pass

            self.step += 1
            # if self.step % self.lr_decay_steps == 0:
            #     self.scheduler.step()
            if self.step % self.eval_per_steps == 0:
                metric = self.eval_model_vqvae()
                
                if metric[self.metric] >= self.best_metric[self.metric]:
                    self.model.eval()
                    torch.save(self.model.state_dict(), self.save_path + '/model.pkl')
                    print(f"保存最佳模型 (Step {self.step})")
                    self.result_file = open(self.save_path + '/result.txt', 'a+')
                    print(f'保存模型 Step {self.step}', file=self.result_file)
                    self.result_file.close()
                    self.best_metric = metric
                self.model.train()

        return loss_sum / (idx + 1), metric, time.perf_counter() - t0

    def eval_model_vqvae(self):
        self.model.eval()
        pbar = tqdm(self.test_loader,
                   desc="评估进度",
                   disable=not self.verbose,
                   ncols=100)

        with torch.no_grad():
            total_correct = 0
            total_f1 = 0
            num_samples_processed = 0  # 用于记录已处理的样本总数

            for idx, batch in enumerate(pbar):
                data = [tensor.to(self.device) for tensor in batch]
                if len(data)!= 2:
                    print(f"Warning: Batch should contain two elements (sequences and targets), but got {len(data)}. Skipping this batch.")
                    continue  # 如果批次数据不符合要求，跳过当前批次

                seqs, target = data
                if not isinstance(seqs, torch.Tensor) or not isinstance(target, torch.Tensor):
                    print("Error: Input sequences or target labels are not in the correct tensor format. Skipping this batch.")
                    continue  # 如果序列或目标不是张量类型，跳过当前批次

                output = self.model(seqs, None, None, None)
                if output.dim() < 2:
                    print("Error: Model output should have at least two dimensions. Skipping this batch.")
                    continue  # 如果模型输出维度不符合要求，跳过当前批次

                pred = output.argmax(dim=1, keepdim=True)
                # 确保pred和target的形状在比较时是兼容的
                if pred.shape!= target.shape:
                    target = target.view(pred.shape)

                correct = pred.eq(target).sum().item()
                total_correct += correct
                num_samples_processed += pred.numel()  # 更新已处理的样本数量

                # 计算F1Score，确保数据类型转换和形状适配正确
                pred_labels = pred.cpu().numpy().flatten()
                target_labels = target.cpu().numpy().flatten()
                if len(pred_labels)!= len(target_labels):
                    print("Error: Predicted labels and target labels have different lengths. Skipping F1 score calculation for this batch.")
                    continue  # 如果预测标签和目标标签长度不一致，跳过当前批次F1分数计算

                try:
                    f1 = f1_score(target_labels, pred_labels, average='macro')
                    total_f1 += f1
                except ValueError as e:
                    print(f"Error calculating F1 score: {str(e)}. Skipping this batch's F1 score calculation.")
                    continue  # 如果F1分数计算出现异常，跳过当前批次F1分数计算

                pbar.set_postfix({"Accuracy": f"{total_correct / num_samples_processed:.4f}", "F1 Score": f"{total_f1 / (idx + 1):.4f}"})

            if num_samples_processed > 0:  # 避免除数为0，只有当有数据参与循环时才进行平均计算
                metrics = {'acc': total_correct / num_samples_processed, 'F1': total_f1 / (idx + 1)}
            else:
                print("Warning: No valid samples were processed. Returning default metrics values.")
                metrics = {'accuracy': 0.0, 'F1': 0.0}  # 如果没有处理有效样本，返回默认的评估指标值

        return metrics


class RayTrainer(Trainer):
    def __init__(self, args, model, train_loader, test_loader, verbose=False):
        args.epoch = False
        super().__init__(args, model, train_loader, test_loader, verbose=False)

    def train(self):

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        for epoch in range(self.num_epoch):
            loss_epoch, acc_epoch, time_cost = self._train_one_epoch(epoch)

            # tune.report()
            train.report(
            {
            "accuracy":acc_epoch,
            "loss":loss_epoch
            },  # 这里假设你在训练过程中能计算得到current_accuracy_value这个准确率的值
            checkpoint=None)
            
        print(f"训练完成! 最佳指标: {self.best_metric:.4f}")
        return self.best_metric, loss_epoch
    

class WandBTrainer(Trainer):
    def __init__(self, args, model, train_loader, test_loader, verbose=False):
        args.epoch = False
        super().__init__(args, model, train_loader, test_loader, verbose=False)
        
    def train(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        for epoch in range(self.num_epoch):
            loss_epoch, metric, time_cost = self._train_one_epoch(epoch)
            acc_epoch, F1_epoch = metric["accuracy"], metric["F1"]
            wandb.log({"epoch": epoch, "loss": loss_epoch, "accuracy": acc_epoch})
               
        print(f"训练完成! 最佳指标: acc:{self.best_metric[0]:.4f} F1:{self.best_metric[1]:.4f}")

        # 添加测试代码
        self.model.load_state_dict(torch.load(self.save_path + '/model.pkl'))
        self.model.eval()
        test_metric, _ = self.eval_model_vqvae()
        test_acc, test_F1 = test_metric["accuracy"], test_metric["F1"]
        wandb.log({"test/accuracy": test_acc, "test/F1": test_F1})
        print(f"测试完成! 测试指标: acc:{test_acc:.4f} F1:{test_F1:.4f}")
        
        return self.best_metric, loss_epoch
