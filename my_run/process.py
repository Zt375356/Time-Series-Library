import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from utils import *
from tqdm import tqdm

class LossCalculator:
    def __init__(self, model, latent_loss_weight=0.25):
        self.model = model
        self.latent_loss_weight = latent_loss_weight
        self.mse = nn.MSELoss()

    def compute(self, batch):
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
        self.model_name = self.model.get_name()
        print(f"模型名称: {self.model_name}")

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
            data = batch[0].to(self.device)
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
        metrics = {'mse': 0}

        with torch.no_grad():
            for idx, batch in enumerate(pbar):
                data = batch[0].to(self.device)
                mse = self.cr.compute(data)
                metrics['mse'] -= mse
                pbar.set_postfix({"MSE": f"{-metrics['mse']/(idx+1):.4f}"})
                
        metrics['mse'] /= (idx + 1)
        return metrics

    def print_process(self, *x):
        if self.verbose:
            print(*x)