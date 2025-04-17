from torch_geometric.nn.pool.select.topk import topk
from torch_geometric.nn.pool.connect.filter_edges import filter_adj
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import  GATConv

class SAGPooling(nn.Module):
    def __init__(self, in_channels, ratio, conv_type='gat', heads=2):
        """
        SAGPooling 层，增加了特征提取和注意力机制的灵活性。
        """
        super(SAGPooling, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.conv_type = conv_type
        self.heads = heads

        self.conv = GATConv(in_channels, in_channels // heads, heads=heads)

        # 初始化自注意力的权重
        self.attention_weights = nn.Parameter(torch.Tensor(in_channels, 1))
        torch.nn.init.kaiming_uniform_(self.attention_weights)

    def forward(self, x, edge_index, batch):
        """
        前向传播函数
        """
        # Step 1: 图卷积提取节点特征
        x = F.relu(self.conv(x, edge_index))

        # Step 2: 自注意力得分计算
        attention_scores = torch.matmul(x, self.attention_weights).squeeze(-1)
        attention_scores = F.tanh(attention_scores)

        # Step 3: 选择得分最高的前 K% 节点
        perm = topk(attention_scores, self.ratio, batch)

        # Step 4: 根据选定节点的索引进行池化
        x_pool = x[perm]
        batch_pool = batch[perm]
        # 过滤池化后的边
        edge_index_pool, _ = filter_adj(edge_index, None, perm, num_nodes=x.size(0))

        return x_pool, edge_index_pool, batch_pool,perm
