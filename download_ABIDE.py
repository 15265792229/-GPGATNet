import os
import argparse
import pandas as pd
from construct_graph import brain_graph
from nilearn.datasets import fetch_abide_pcp
import requests
def download_file(src_path, info_path):
    try:
        # 发送GET请求以获取文件
        response = requests.get(src_path, stream=True)  # 发起GET请求，stream=True表示以流的形式接收响应
        response.raise_for_status()  # 如果请求出错，这里会抛出HTTPError异常

        # 使用'wb'模式打开文件，用于写入二进制数据
        with open(info_path, 'wb') as file:  # 以二进制写模式打开文件
            for chunk in response.iter_content(chunk_size=8192):  # 分块读取响应内容
                # 如果chunk不为空，则写入文件
                if chunk:
                    file.write(chunk)  # 将读取的数据块写入文件

        print(f"文件已成功保存到 {info_path}")  # 打印文件保存成功的消息
    except requests.RequestException as e:
        print(f"下载文件时发生错误: {e}")  # 捕获并打印下载过程中发生的异常


def load_text(data_path, text):
    # 列出data_path目录下所有以.1D结尾的文件
    files = [f for f in os.listdir(data_path) if f.endswith('.1D')]

    # 移除文件名中的.1D后缀，获取纯文件名
    filenames = [name.split('.')[0] for name in files]  # 移除.1D

    # 进一步处理文件名，移除文件名末尾的固定部分（如_rois_ho, _rois_cc200 等后缀）
    file_idx = [name.rsplit('_', 2)[0] for name in filenames]  # 移除最后一个下划线及后缀

    # 创建一个DataFrame，包含处理后的文件名作为索引（FILE_ID）和原始文件名
    idx = pd.DataFrame({'FILE_ID': file_idx, 'file_name': files})

    # 使用'FILE_ID'作为键，将idx DataFrame与text DataFrame进行左连接（left join）
    # 假设text DataFrame中也包含一个名为'FILE_ID'的列，用于匹配
    logs = pd.merge(idx, text, how='left', on='FILE_ID')

    # 返回合并后的DataFrame，其中包含所有下载样本的非成像信息
    return logs


if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data', help='存储脑图数据的路径')  # 添加命令行参数--root，用于指定存储数据的根目录
    parser.add_argument('--derivatives',default='rois_ho',type=str, help='存储脑图数据的路径')  # 添加命令行参数--root，用于指定存储数据的根目录
    parser.add_argument('--verbose', type=bool, default=True, help='是否打印下载详情')  # 添加命令行参数--verbose，用于控制是否打印下载详情
    args = parser.parse_args()  # 解析命令行参数
    # 打印正在下载ABIDE I数据集（由CPAC预处理）的信息
    print('正在下载由CPAC预处理的ABIDE I数据集...')
    # 调用函数下载数据集，设置数据目录、导出物类型、详细程度等参数
    fetch_abide_pcp(data_dir=args.root, derivatives=args.derivatives, verbose=args.verbose,
                    pipeline='cpac', band_pass_filtering=True, global_signal_regression=True)

    # 由fetch abide生成的路径
    path = os.path.join('./data', 'ABIDE_pcp', 'cpac', 'filt_global')

    # 表型信息路径
    info_path = os.path.join(args.root, 'phenotypic')
    # 如果表型信息路径不存在，则创建它
    if not os.path.exists(info_path):
        os.makedirs(info_path)
        # 打印正在加载表型信息
    print('正在加载表型信息')
    # 读取表型信息CSV文件
    phenotypic = pd.read_csv(os.path.join('./data', 'ABIDE_pcp', 'Phenotypic_V1_0b_preprocessed1.csv'))
    # 调用load_text函数加载非成像信息，并与表型信息合并
    logs = load_text(path, phenotypic)

    # 重设标签值
    logs['label'] = [2 - i for i in logs['DX_GROUP']]
    # 将合并后的数据保存到CSV文件
    logs.to_csv(os.path.join(args.root, 'phenotypic', 'log.csv'))

    print('处理数据...')
    brain_graph(logs, os.path.join(args.root, 'ABIDE', 'raw'), path)

