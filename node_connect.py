import argparse
import os


def node_connect(num_samples, num_nodes_per_sample):
    # 确保目录存在
    file_path = 'data/ABIDE/raw/ABIDE_A.txt'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 打开文件用于保存结果
    with open(file_path, 'w') as f:
        # 遍历每个样本
        for sample_index in range(num_samples):
            # 计算当前样本的节点范围
            start_node = sample_index * num_nodes_per_sample + 1
            end_node = start_node + num_nodes_per_sample - 1

            # 遍历当前样本中的每个节点
            for node1 in range(start_node, end_node + 1):
                for node2 in range(start_node, end_node + 1):
                    if node1 != node2:  # 确保不自连
                        # 写入文件，每个连线一行
                        f.write(f"{node1},{node2}\n")


if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description='Generate node connections for a given number of samples and nodes per sample.')

    # 添加命令行参数
    parser.add_argument('--num_samples', type=int, required=True, help='Number of samples')
    parser.add_argument('--num_nodes_per_sample', type=int, required=True, help='Number of nodes per sample')

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数并传入参数
    node_connect(args.num_samples, args.num_nodes_per_sample)
