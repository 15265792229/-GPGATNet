import os
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial.distance import cosine
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

def gennoimggraph():
    data_dir = './data'
    cluster_att = ['SEX', 'SITE_ID']
    # 获取文本信息：性别、站点
    logs = pd.read_csv(os.path.join(data_dir, 'phenotypic', 'log.csv'))
    text_info = logs[cluster_att].values
    enc = OneHotEncoder()  # 独热编码
    enc.fit(text_info)
    text_feature = enc.transform(text_info).toarray()  # 转换为独热编码的数组

    # 考虑年龄
    ages = logs['AGE_AT_SCAN'].values
    # 标准化年龄
    ages = (ages - min(ages)) / (max(ages) - min(ages))

    # 将文本特征和年龄合并
    cluster_features = np.c_[text_feature, ages]

    adj = []  # 邻接列表
    att = []  # 边权重列表
    sim_matrix = cosine_similarity(cluster_features)  # 计算余弦相似度矩阵
    for i in range(871):
        for j in range(871):
            if sim_matrix[i, j] > 0.5 and i > j:  # 只考虑i>j的情况，避免重复边
                adj.append([i, j])
                att.append(sim_matrix[i, j])

    adj = np.array(adj).T  # 转置为两列矩阵
    att = np.array([att]).T  # 转置为两列矩阵（虽然这里可能是多余的，但保持格式一致）

    # 如果目标文件夹不存在，则创建
    if not os.path.exists(os.path.join(data_dir, 'population graph')):
        os.makedirs(os.path.join(data_dir, 'population graph'))

        # 保存邻接矩阵和边权重到CSV文件
    pd.DataFrame(adj).to_csv(os.path.join(data_dir, 'population graph', 'graphnoimg.adj'), index=False, header=False)
    pd.DataFrame(att).to_csv(os.path.join(data_dir, 'population graph', 'graphnoimg.attr'), index=False, header=False)

def build_population_graph(data_dir, threshold):
    """
    根据功能连接矩阵构建人群图，使用余弦相似度，避免重复边。
    :param data_dir: 保存功能连接矩阵的文件夹路径，每个文件代表一个个体的功能连接矩阵
    :param threshold: 相似度阈值，大于该阈值的相似度用于构建边
    :return: edge_index, edge_attr
    """
    files = sorted(os.listdir(data_dir))  # 获取文件列表，代表所有个体的功能连接矩阵
    num_samples = len(files)

    # 存储所有个体的功能连接矩阵
    matrices = []
    for file in files:
        file_path = os.path.join(data_dir, file)
        matrix = np.loadtxt(file_path)  # 加载功能连接矩阵
        matrices.append(matrix)

    matrices = np.stack(matrices)  # [num_samples, num_features], 每个个体的功能连接矩阵

    # 计算个体之间的相似性（使用余弦相似度）
    similarity_matrix = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(i + 1, num_samples):  # 保证只计算 i < j 的情况，避免重复边
            cos_sim = 1 - cosine(matrices[i], matrices[j])  # 余弦相似度 = 1 - 余弦距离
            enhanced_sim = np.exp(cos_sim) - 1
            similarity_matrix[i, j] = enhanced_sim
            similarity_matrix[j, i] = enhanced_sim

    # 构建边索引和边权重，确保只保留 i < j 的边
    edge_index = []
    edge_attr = []

    for i in range(num_samples):
        for j in range(i + 1, num_samples):  # 确保 i < j 来避免重复边
            if similarity_matrix[i, j] > threshold:
                edge_index.append([i, j])
                edge_attr.append(similarity_matrix[i, j])

    # 将 edge_index 和 edge_attr 转换为张量
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # (2, num_edges)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return edge_index, edge_attr



def save_graph_to_file(edge_index, edge_attr, adj_file, attr_file):
    """
    将边索引和边权重保存到文件，边索引保存为两行，逗号分隔。
    :param edge_index: 边索引 (2, num_edges)
    :param edge_attr: 边权重 (num_edges,)
    :param adj_file: 边索引保存的文件名
    :param attr_file: 边权重保存的文件名
    """
    # 创建保存目录
    os.makedirs('data/population graph', exist_ok=True)

    # 更新文件路径
    adj_file = os.path.join('data/population graph', adj_file)
    attr_file = os.path.join('data/population graph', attr_file)

    # 转换 edge_index 为 NumPy 数组并分开为两行
    edge_index_np = edge_index.numpy()
    start_nodes = edge_index_np[0]  # 第一行，起始点
    end_nodes = edge_index_np[1]  # 第二行，目的节点

    # 保存起始点和目的节点，每行用逗号分隔
    with open(adj_file, 'w') as f:
        f.write(','.join(map(str, start_nodes)) + '\n')  # 写入起始点
        f.write(','.join(map(str, end_nodes)) + '\n')  # 写入目的节点

    # 保存边权重
    np.savetxt(attr_file, edge_attr, fmt='%.6f')  # 保存为浮点数


def mixgraph():
    data_dir = './data'
    adj_path = os.path.join(data_dir, 'population graph', 'graphcos.adj')
    attr_path = os.path.join(data_dir, 'population graph', 'graphcos.attr')
    adj_path2 = os.path.join(data_dir, 'population graph', 'graphnoimg.adj')
    attr_path2 = os.path.join(data_dir, 'population graph', 'graphnoimg.attr')

    # 加载人群图，edge_index 表示边索引，edge_attr 表示边权重
    edge_index1 = pd.read_csv(adj_path, header=None).values
    edge_attr1 = pd.read_csv(attr_path, header=None).values.reshape(-1)

    edge_index2 = pd.read_csv(adj_path2, header=None).values
    edge_attr2 = pd.read_csv(attr_path2, header=None).values.reshape(-1)

    # 创建新的边索引矩阵和边权重矩阵，初始化为与edge_index2和edge_attr2相同
    new_edge_index = edge_index2.copy()
    new_edge_attr = edge_attr2.copy()

    # 用于查找edge_index1中的边是否存在于edge_index2中，返回边索引位置或-1
    def find_edge_index_in_edge_index2(edge, edge_index2):
        # 返回边在edge_index2中的索引，如果不存在则返回-1
        match = np.where((edge_index2[:, 0] == edge[0]) & (edge_index2[:, 1] == edge[1]))[0]
        return match[0] if len(match) > 0 else -1

    # 遍历edge_index1和edge_attr1，如果边在edge_index2中存在，则更新权重
    for i in range(edge_index1.shape[0]):
        edge = edge_index1[i]  # 取出边
        weight1 = edge_attr1[i]  # 取出边权重1

        idx_in_edge_index2 = find_edge_index_in_edge_index2(edge, edge_index2)

        if idx_in_edge_index2 != -1:
            # 如果边在edge_index2中存在，取出对应的权重
            weight2 = edge_attr2[idx_in_edge_index2]
            # 进行加权平均
            new_edge_attr[idx_in_edge_index2] = (weight1 + weight2) / 2

    # 保存融合后的图为新的文件
    new_adj_path = os.path.join(data_dir, 'population graph', 'mixgraph.adj')
    new_attr_path = os.path.join(data_dir, 'population graph', 'mixgraph.attr')

    # 保存更新后的边索引和边权重
    np.savetxt(new_adj_path, new_edge_index, fmt='%d', delimiter=',')
    np.savetxt(new_attr_path, new_edge_attr, fmt='%.6f', delimiter=',')
def genpopulation():
    # 假设功能连接矩阵保存在 'dataconnect' 文件夹中，每个文件表示一个个体的功能连接矩阵
    data_dir = './data/dataconnect'

    # 构建图，生成边索引和边权重
    edge_index, edge_attr = build_population_graph(data_dir, threshold=0.1)

    # 保存边索引和边权重到文件
    save_graph_to_file(edge_index, edge_attr, adj_file='graphcos.adj', attr_file='graphcos.attr')


if __name__ == '__main__':
    gennoimggraph()
    genpopulation()
    mixgraph()
