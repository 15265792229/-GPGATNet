from torch_geometric.utils import dropout_edge

from layers import SAGPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import SAGEConv

class SAGNet(torch.nn.Module):
    def __init__(self,args):
        super(SAGNet, self).__init__()
        self.pooling_ratio = args.pooling_ratio
        self.in_channels = None  # 初始化时暂不设置in_channels

        # 三层 SAGPooling，每一层都有池化操作
        # 注意：初始化SAGPooling时不能提前设定in_channels
        self.pool1 = None
        self.pool2 = None
        self.pool3 = None

        # 用于融合的线性层，output_dim 是最终输出维度，可以根据任务设定
        self.fc = None  # 同样暂不初始化线性层
        self.output_dim = 376

        # 新增 Dropout 层，概率可以根据需要进行调整
        self.dropout = torch.nn.Dropout(p=self.pooling_ratio)

    def forward(self, data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 动态设置in_channels
        if self.in_channels is None:
            num_features = (data.x[0] != 0).sum().item()
            if num_features % 2 == 0:
                self.in_channels = num_features
            else:
                self.in_channels = num_features + 1
                zero_padding = torch.zeros(data.x.size(0), 1, device=data.x.device)
                data.x = torch.cat([data.x, zero_padding], dim=1)

            # 使用动态设置的in_channels初始化SAGPooling和全连接层
            self.pool1 = SAGPooling(self.in_channels, self.pooling_ratio).to(device)
            self.pool2 = SAGPooling(self.in_channels, self.pooling_ratio).to(device)
            self.pool3 = SAGPooling(self.in_channels, self.pooling_ratio).to(device)
            self.fc = torch.nn.Linear(self.in_channels * 2 * 3, self.output_dim).to(device)
        x, edge_index, batch= data.x[:, :self.in_channels], data.edge_index, data.batch
        # 第一层池化
        x, edge_index, batch,perm1 = self.pool1(x, edge_index, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x1 = self.dropout(x1)
        # 第二层池化
        x, edge_index, batch,_ = self.pool2(x, edge_index, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x2 = self.dropout(x2)
        # 第三层池化
        x, edge_index, batch,_ = self.pool3(x, edge_index, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x3 = self.dropout(x3)

        # 拼接各层特征
        x_concat = torch.cat([x1, x2, x3], dim=1)
        print(x)
        # Dropout 和 relu 激活函数
        x = F.relu(self.fc(x_concat))

        return x


class MH_GCAT(torch.nn.Module):
    def __init__(self, args):
        super(MH_GCAT, self).__init__()
        self.num_features = 376
        self.nhid = 512
        self.dropout_ratio =args.dropout_ratio
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.conv3 = SAGEConv(self.nhid, self.nhid)

        # 定义两个不同注意力头数的GAT卷积层
        self.attention_conv1 = GATConv(self.nhid*3, 1, heads=4, concat=False)

        self.attention_conv2 = GATConv(self.nhid*3, 1, heads=16, concat=False)
    def forward(self, x, edge_index, edge_weight):

        x1 = self.conv1(x, edge_index, edge_weight)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout_ratio, training=self.training)

        x2 = self.conv2(x1, edge_index, edge_weight)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=self.dropout_ratio, training=self.training)

        x3 = self.conv3(x1, edge_index)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, p=self.dropout_ratio, training=self.training)

        dense_features = torch.cat([x1,x2,x3], dim=1)
        dense_features = F.dropout(dense_features, p=0.1, training=self.training)
        edge_index, _ = dropout_edge(edge_index=edge_index,p=0.1,training=self.training)
        # 分别通过两个注意力卷积层
        attention_out1 = self.attention_conv1(dense_features, edge_index)
        attention_out2 = self.attention_conv2(dense_features, edge_index)

        # 融合两个注意力卷积的输出
        x_out = (attention_out1+attention_out2)/2
        x_out = torch.flatten(x_out)
        return x_out, dense_features
