import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity


def brain_graph(logs, path, data_folder):
    if not os.path.exists(path):
        os.makedirs(path)  # 如果目标路径不存在，则创建它

    print('正在处理图形标识符...')
    indicator = np.repeat(np.arange(1, 872), 111)  # 创建一个由1到871的数字，每个数字重复111次
    pd.DataFrame(indicator).to_csv(os.path.join(path, 'ABIDE_graph_indicator.txt'), index=False,
                                   header=False)  # 保存到CSV文件
    print('完成!')

    print('正在处理图形标签...')
    graph_labels = logs[['label']]  # 从logs DataFrame中提取标签列
    graph_labels.to_csv(os.path.join(path, 'ABIDE_graph_labels.txt'), index=False, header=False)  # 保存到CSV文件
    print('完成!')

    print('正在处理节点属性...')
    # 遵循log.csv中的顺序
    files = logs['file_name']  # 从logs DataFrame中获取文件名
    node_att = pd.DataFrame([])  # 初始化一个空的DataFrame来存储节点属性
    for file in files:
        file_path = os.path.join(data_folder, file)  # 构建文件路径
        # 来自不同站点的数据可能具有不同的时间长度（数据文件中的行数）
        ho_rois = pd.read_csv(file_path, sep='\t').iloc[:300, :].T  # 读取并转置前300行，数据的长度都不到300，所以取300一定能保存所有数据
        node_att = pd.concat([node_att, ho_rois])  # 将读取的节点属性追加到node_att中

    node_att.to_csv(os.path.join(path, 'ABIDE_node_attributes.txt'), index=False, header=False)  # 保存到CSV文件
    node_attributes_path = './data/ABIDE/raw/ABIDE_node_attributes.txt'
    df = pd.read_csv(node_attributes_path, header=None)

    # 用0填充空值
    df = df.replace('', 0).replace(np.nan, 0)#防止空值影响，0不需要，之后会处理

    # 保存修正后的文件
    cleaned_path = './data/ABIDE/raw/ABIDE_node_attributes.txt'
    df.to_csv(cleaned_path, index=False, header=False)
    print('节点属性处理完成!')

    print('正在处理节点标签...')
    cols = list(pd.read_csv(file_path, sep='\t').columns.values)  # 读取一个文件以获取列名作为标准
    for file in files:
        file_path = os.path.join(data_folder, file)
        temp_cols = list(pd.read_csv(file_path, sep='\t').columns.values)
        assert cols == temp_cols, 'ABIDE pcp中脑区顺序不一致！'  # 验证其他文件列名是否与标准一致

    # 创建节点标签，从1到111，然后重复871次
    node_label = np.arange(111)
    node_labels = np.tile(node_label, 871)
    pd.DataFrame(node_labels).to_csv(os.path.join(path, 'ABIDE_node_labels.txt'), index=False, header=False)
    print('完成!')


