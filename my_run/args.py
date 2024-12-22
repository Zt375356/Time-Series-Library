import argparse
import os
import json

parser = argparse.ArgumentParser()

# 数据集和数据加载器参数

parser.add_argument('--task_name', type=str, default='classification')
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--model', type=str, default='TimesNet')
parser.add_argument('--des', type=str, default='Exp')
parser.add_argument('--itr', type=int, default=1)


parser.add_argument('--save_path', type=str, default='./log')
parser.add_argument('--dataset_name', type=str, default='HAR-A', choices=['AW-A', 'AW-B', 'Gesture-A', 'Gesture-B', 'HAR-A', 'HAR-B', 'HAR-C'])
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--useLeaveOneOutCrossValidation', type=bool, default=0)


# model args
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--d_ff', type=int, default=64)
parser.add_argument('--embed',type=str,default='fixed',choices=['fixed','timeF'])
parser.add_argument('--freq',type=str,default='h')
parser.add_argument('--top_k', type=int, default=3)
parser.add_argument('--num_kernels',type=int,default=3)
parser.add_argument('--dropout', type=float, default=0.1)


# train args
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--lr_decay_rate', type=float, default=0.99)
parser.add_argument('--lr_decay_steps', type=int, default=300)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--eval_per_steps', type=int, default=200)



args = parser.parse_args()
args.save_path = args.save_path+"/"+args.dataset_name

if args.save_path == 'None':
    path_str = 'D-' + str(args.d_model) + '_Model-' + args.model + '_Lr-' + str(args.lr) + '_Dataset-' + args.dataset + '/'
    args.save_path = path_str
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path, exist_ok=True)

config_file = open(args.save_path + '/args.json', 'w')
tmp = args.__dict__
json.dump(tmp, config_file, indent=1)
print(args)
config_file.close()