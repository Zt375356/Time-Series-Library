import argparse
import os
import json

parser = argparse.ArgumentParser()


# 数据集和数据加载器参数
parser.add_argument('--save_path', type=str, default='./ecg_tokenizer')
parser.add_argument('--dataset_name', type=str, default='HAR-A', choices=['AW-A', 'AW-B', 'Gesture-A', 'Gesture-B', 'HAR-A', 'HAR-B', 'HAR-C'])
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--useLeaveOneOutCrossValidation', type=bool, default=0)

# model args
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--n_embed', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--wave_length', type=int, default=32)

parser.add_argument('--eval_per_steps', type=int, default=300)
parser.add_argument('--enable_res_parameter', type=int, default=1)
parser.add_argument('--pooling_type', type=str, default='mean', choices=['mean', 'max', 'last_token'])

# tcn args
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--block_num', type=int, default=4)
parser.add_argument('--dilations', type=list, default=[1, 4])

# train args
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--lr_decay_rate', type=float, default=0.99)
parser.add_argument('--lr_decay_steps', type=int, default=300)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--num_epoch', type=int, default=100)

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
# print(args)
config_file.close()