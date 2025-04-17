from training import  train_gcn, test_gcn
from models import  MH_GCAT
import pandas as pd
import numpy as np
import os
import glob
import torch
from torch_geometric.data import DataLoader, Data
from sklearn.model_selection import KFold

def kfold_MH_GCAT(edge_index, edge_attr, num_samples, args):
    # 局部设置参数
    args.num_features = 376# MLP的输出特征大小
    args.nhid = 512# 隐藏层特征大小
    args.epochs = 30000  # 最大训练轮次
    args.patience = 5000  # 早期停止的耐心值，基于验证集上的性能
    args.weight_decay = 0.001  # 权重衰减
    args.least = 0  # 最小训练轮次

    # 加载人口图
    edge_index = torch.tensor(edge_index, dtype=torch.long)  # 将边索引转换为torch张量
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # 将边权重转换为torch张量

    # 生成样本索引
    indices = np.arange(num_samples)  # 生成从0到num_samples-1的索引数组

    # 初始化K折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=args.seed)  # 10折交叉验证，打乱数据，设置随机种子

    # 存储预测结果
    result_df = pd.DataFrame([])  # 初始化一个空的DataFrame用于存储预测结果
    test_result_acc = []  # 初始化列表用于存储测试集上的准确率
    test_result_loss = []  # 初始化列表用于存储测试集上的损失


    for i, (train_idx, test_idx) in enumerate(kf.split(indices)):
        fold_path = os.path.join(args.data_dir, 'Features', 'fold_%d' % (i + 1))
        # GCN训练的工作路径
        work_path = os.path.join(args.check_dir, 'GCN')
        print(test_idx)
        # 打乱训练索引，以随机分配验证集和测试集（无嵌套搜索）
        np.random.shuffle(train_idx)
        val_idx = train_idx[:len(train_idx) // 10]  # 取前10%作为验证集
        train_idx = train_idx[len(train_idx) // 10:]  # 剩余作为训练集

        # 确保三个数据集（训练集、测试集、验证集）是独立的
        assert len(set(list(train_idx) + list(test_idx) + list(val_idx))) == num_samples, \
            '交叉验证中有问题，数据集可能不是完全独立的'

        # 如果工作路径不存在，则创建它
        if not os.path.exists(work_path):
            os.makedirs(work_path)

        print('正在第%d折上训练GCN' % (i + 1))
        # 初始化GCN模型并移动到指定的设备上
        model = MH_GCAT(args).to(args.device)
        # 初始化优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # 加载'进一步学习特征'
        feature_path = os.path.join(fold_path, 'features.txt')
        assert os.path.exists(feature_path), \
            '未找到进一步学习特征文件！'
        content = pd.read_csv(feature_path, header=None, sep='\t')  # 读取特征文件

        # 分离特征和标签
        x = content.iloc[:, :-1].values  # 特征
        y = content.iloc[:, -1].values  # 标签

        # 将numpy数组转换为torch张量
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)

        # 创建图数据对象
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        # 根据索引生成掩码
        train_mask = np.zeros(num_samples)
        test_mask = np.zeros(num_samples)
        val_mask = np.zeros(num_samples)
        train_mask[train_idx] = 1
        test_mask[test_idx] = 1
        val_mask[val_idx] = 1

        # 设置数据集的掩码
        data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
        data.val_mask = torch.tensor(val_mask, dtype=torch.bool)

        # 确保掩码之间没有重叠！
        # 在实验中这是必要的
        assert np.array_equal(train_mask + val_mask + test_mask, np.ones_like(train_mask)), \
            '交叉验证中存在问题！'

        # 由于这里只有一个图数据对象，所以批大小没有意义
        loader = DataLoader([data], batch_size=1)

        # 模型训练
        best_model = train_gcn(loader, model, optimizer, work_path, args)
        # 为测试集恢复最佳模型
        checkpoint = torch.load(os.path.join(work_path, '{}.pth'.format(best_model)))
        model.load_state_dict(checkpoint['net'])
        test_acc, test_loss, test_out = test_gcn(loader, model, args)

        # 存储结果
        result_df['fold_%d_' % (i + 1)] = test_out
        test_result_acc.append(test_acc)
        test_result_loss.append(test_loss)
        # 在验证集上进行测试（注意这里test参数设置为False）
        acc_val, loss_val, _ = test_gcn(loader, model, args, test=False)
        print('GCN 第{:0>2d}折测试集结果，损失 = {:.6f}，准确率 = {:.6f}'.format(i + 1, test_loss, test_acc))
        print('GCN 第{:0>2d}折验证集结果，损失 = {:.6f}，准确率 = {:.6f}'.format(i + 1, loss_val, acc_val))

        # 保存模型状态
        state = {'net': model.state_dict(), 'args': args}
        torch.save(state, os.path.join(work_path, 'fold_{:d}_test_{:.6f}_drop_{:.3f}_epoch_{:d}_.pth'
                                       .format(i + 1, test_acc, args.dropout_ratio, best_model)))

    # 将预测结果保存到 args.result_dir/Graph Convolutional Networks/GCN_pool_%.3f_seed_%d_.csv
    # 其中 %.3f 是池化比例（保留三位小数），%d 是随机种子
    result_path = args.result_dir
    # 如果结果目录不存在，则创建它
    if not os.path.exists(result_path):
        os.makedirs(result_path)

        # 将结果DataFrame保存到CSV文件，文件名包含池化比例和随机种子
    result_df.to_csv(os.path.join(result_path,
                                  'GCN_pool_%.3f_seed_%d_.csv' % (args.pooling_ratio, args.seed)),
                     index=False, header=True)  # 不保存索引，包含表头

    # 打印平均准确率
    print('平均准确率: %f' % (sum(test_result_acc) / len(test_result_acc)))



def load_all_folds_and_evaluate(edge_index, edge_attr, num_samples, args):
    args.num_features = 376  # MLP的输出特征大小
    args.nhid = 512  # 隐藏层特征大小

    """
    加载 10 折交叉验证的最佳模型，并计算每个折的测试准确率与损失，最终得到平均准确率和损失。
    :param edge_index: 人口图的邻接矩阵。
    :param edge_attr: 边权重，例如余弦相似度值。
    :param args: 从 main.py 传入的参数。
    :param num_samples: 样本数量，用于生成掩码。
    :return: 所有折的平均测试准确率和损失。
    """

    # 将 edge_index 和 edge_attr 转换为正确的类型和形状
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # 初始化 K 折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=args.seed)

    # 存储每个折的测试结果
    all_test_acc = []
    all_test_loss = []
    test_results = []

    for fold_num, (train_idx, test_idx) in enumerate(kf.split(np.arange(num_samples)), start=1):
        print(f"加载第 {fold_num} 折的最佳模型并进行测试...")

        # 设置模型文件夹路径并查找以 fold_{fold_num}_test 开头的模型文件
        work_path = os.path.join(args.check_dir, 'GCN')
        model_files = glob.glob(os.path.join(work_path, f'fold_{fold_num}_test_*.pth'))

        if not model_files:
            print(f"第 {fold_num} 折的模型文件未找到，跳过...")
            continue

        model_path = model_files[0]
        print(f"加载模型文件：{model_path}")

        # 初始化模型
        model = MH_GCAT(args).to(args.device)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['net'])

        # 加载 '进一步学习特征'
        fold_path = os.path.join(args.data_dir, 'Features', f'fold_{fold_num}')
        feature_path = os.path.join(fold_path, 'features.txt')
        content = pd.read_csv(feature_path, header=None, sep='\t')

        x = torch.tensor(content.iloc[:, :-1].values, dtype=torch.float)
        y = torch.tensor(content.iloc[:, -1].values, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        # 打乱训练索引，创建验证集和训练集（与 kfold_gcn 保持一致）
        np.random.shuffle(train_idx)
        val_idx = train_idx[:len(train_idx) // 10]
        train_idx = train_idx[len(train_idx) // 10:]

        # 创建掩码
        train_mask = np.zeros(num_samples, dtype=bool)
        val_mask = np.zeros(num_samples, dtype=bool)
        test_mask = np.zeros(num_samples, dtype=bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
        data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

        # 创建数据加载器
        loader = DataLoader([data], batch_size=1)

        # 计算准确率和损失
        test_acc, test_loss, _ = test_gcn(loader, model, args)

        print(f'第 {fold_num} 折的测试集结果：准确率 = {test_acc:.6f}，损失 = {test_loss:.6f}')

        # 将当前折的结果添加到列表中
        all_test_acc.append(test_acc)
        all_test_loss.append(test_loss)
        test_results.append({
            'fold': fold_num,
            'test_accuracy': test_acc,
            'test_loss': test_loss
        })

    # 计算并输出平均准确率和损失
    avg_test_acc = np.mean(all_test_acc)
    avg_test_loss = np.mean(all_test_loss)

    print(f'10 折交叉验证的平均测试准确率 = {avg_test_acc:.6f}，平均测试损失 = {avg_test_loss:.6f}')

    # 将所有折的结果保存到 CSV 文件中
    result_df = pd.DataFrame(test_results)
    result_file_path = os.path.join(args.result_dir, 'kfold_test_results.csv')
    result_df.to_csv(result_file_path, index=False)

    return avg_test_acc, avg_test_loss

