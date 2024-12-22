import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import f1_score
from utils import *
from tqdm import tqdm
import time

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
        if self.num_epoch:
            self.result_file = open(self.save_path + '/result.txt', 'w')
            self.result_file.close()

        self.step = 0
        self.best_metric = -1e9
        self.metric = 'mse'

    def train(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_decay ** step, verbose=self.verbose)
        
        for epoch in range(self.num_epoch):
            loss_epoch, time_cost = self._train_one_epoch()
            self.result_file = open(self.save_path + '/result.txt', 'a+')
            log_msg = f'Epoch [{epoch+1}/{self.num_epoch}] Loss: {loss_epoch:.4f} Time: {time_cost:.2f}s'
            print(log_msg)
            print(log_msg, file=self.result_file)
            self.result_file.close()
            
        print(f"训练完成! 最佳指标: {self.best_metric:.4f}")
        return self.best_metric
    
    def _train_one_epoch(self):
        t0 = time.perf_counter()
        self.model.train()
        
        pbar = tqdm(self.train_loader, 
                   desc="训练进度",
                   disable=not self.verbose,
                   ncols=100)

        loss_sum = 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            data = [tensor.to(self.device) for tensor in batch]
            loss = self.cr.compute(data)
            loss_sum += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            self.step += 1
            if self.step % self.lr_decay_steps == 0:
                self.scheduler.step()
            if self.step % self.eval_per_steps == 0:
                metric = self.eval_model_vqvae()
                print(f"Step {self.step}: {metric}")
                self.result_file = open(self.save_path + '/result.txt', 'a+')
                print(f'Step {self.step}:', file=self.result_file)
                print(metric, file=self.result_file)
                self.result_file.close()
                
                if metric[self.metric] >= self.best_metric:
                    self.model.eval()
                    torch.save(self.model.state_dict(), self.save_path + '/model.pkl')
                    print(f"保存最佳模型 (Step {self.step})")
                    self.result_file = open(self.save_path + '/result.txt', 'a+')
                    print(f'保存模型 Step {self.step}', file=self.result_file)
                    self.result_file.close()
                    self.best_metric = metric[self.metric]
                self.model.train()

        return loss_sum / (idx + 1), time.perf_counter() - t0

    def eval_model_vqvae(self):
        self.model.eval()
        pbar = tqdm(self.test_loader,
                   desc="评估进度",
                   disable=not self.verbose,
                   ncols=100)
        metrics = {'acc': 0, 'F1': 0}

        with torch.no_grad():
            for idx, batch in enumerate(pbar):
                data = [tensor.to(self.device) for tensor in batch]

                seqs, target = data

                output = self.model(seqs, None, None, None)

                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                acc = correct / len(data)
                metrics['acc'] += acc

                # 计算F1Score
                pred_labels = pred.cpu().numpy().flatten()
                target_labels = target.cpu().numpy().flatten()
                f1 = f1_score(target_labels, pred_labels, average='macro')
                metrics['F1'] += f1

                pbar.set_postfix({"Accuracy": f"{metrics['acc']/(idx+1):.4f}", "F1 Score": f"{metrics['F1']/(idx+1):.4f}"})

            if idx > 0:  # 避免除数为0，只有当有数据参与循环时才进行平均计算
                metrics['acc'] /= (idx + 1)
                metrics['F1'] /= (idx + 1)

        return metrics


    def print_process(self, *x):
        if self.verbose:
            print(*x)