import argparse
import torch
from training import SAGNet_pooling
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=14, help='随机种子')
parser.add_argument('--batch_size', type=int, default=1, help='批量大小')
parser.add_argument('--lr', type=float, default=0.001, help='学习率')
parser.add_argument('--pooling_ratio', type=float, default=0.05, help='池化比例')
parser.add_argument('--dropout_ratio', type=float, default=0.001, help='dropout比例')
parser.add_argument('--data_dir', type=str, default='./data', help='所有数据集的根目录')
parser.add_argument('--device', type=str, default='cuda:0', help='指定CUDA设备')
parser.add_argument('--check_dir', type=str, default='./checkpoints', help='保存模型的根目录')
parser.add_argument('--result_dir', type=str, default='./results', help='分类结果的根目录')
parser.add_argument('--verbose', type=bool, default=True, help='打印训练详情')
args = parser.parse_args()
# 设置随机种子
torch.manual_seed(args.seed)

if __name__ == '__main__':
    SAGNet_pooling(args)