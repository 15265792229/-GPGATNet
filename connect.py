import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import argparse

def generate_fc(path, vectorize=True, discard_diagonal=True):
    """
    从4D fMRI数据中生成功能连接矩阵，使用余弦相似度

    :param path: 4D fMRI数据的文件路径
    :param vectorize: 是否将矩阵转换为向量形式
    :param discard_diagonal: 是否丢弃矩阵的对角线元素
    :return: 功能连接矩阵
    """
    # 从文件路径加载数据为numpy数组
    arr = np.loadtxt(path)  # 加载fMRI数据文件为numpy数组

    # 获取脑区的数量
    n_regions = arr.shape[1]  # 获取时间序列的列数，假设列表示脑区

    # 计算余弦相似度矩阵
    fc_matrix = cosine_similarity(arr.T)  # 转置后每一列表示一个脑区

    # 丢弃对角线元素
    if discard_diagonal:
        np.fill_diagonal(fc_matrix, 0)

    if vectorize:
        # 将矩阵转换为向量（去掉对角线）
        triu_indices = np.triu_indices(n_regions, k=1)
        fc_vector = fc_matrix[triu_indices]
        return fc_vector
    else:
        return fc_matrix

def geconnect():
    parser = argparse.ArgumentParser(
        description='从atlas fMRI数据中生成功能连接矩阵')  # 创建命令行参数解析器并描述其功能
    parser.add_argument('--path', type=str, default='./data/ABIDE_pcp/cpac/filt_global', help='fMRI数据的路径')
    parser.add_argument('--output', type=str, default='./data/dataconnect', help='输出功能连接矩阵的目录路径')  # 添加输出目录参数
    args = parser.parse_args()  # 解析命令行参数

    path = Path(args.path)  # 将输入的路径转换为Path对象
    output = Path(args.output)  # 将输出路径转换为Path对象
    if not output.exists():  # 如果输出目录不存在
        output.mkdir(parents=True, exist_ok=True)  # 创建目录及其父目录，如果已存在则忽略

    files = path.glob('*.1D')  # 在指定路径下查找所有.1D文件
    for file in files:  # 遍历找到的文件
        fc = generate_fc(str(file))  # 为每个文件生成功能连接矩阵，注意将Path对象转换为字符串
        np.savetxt(output.joinpath(file.name), fc, fmt='%.4f')  # 保存功能连接矩阵到指定输出目录，保留4位小数

if __name__ == '__main__':
    geconnect()
