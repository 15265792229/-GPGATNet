import os
import torch
import time
import pandas as pd
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from models import SAGNet
import torch.nn as nn

def SAGNet_pooling(args):
    torch.manual_seed(args.seed)  # 设置随机种子
    # 加载数据
    abide_dataset = TUDataset(args.data_dir, name='ABIDE', use_node_attr=True)  # 加载ABIDE数据集
    args.num_classes = abide_dataset.num_classes  # 设置类别数

    # 运行图池化
    abide_loader = DataLoader(abide_dataset, batch_size=args.batch_size, shuffle=False)  # 创建数据加载器
    downsample = []  # 用于存储池化后的数据
    label = []  # 用于存储标签

    for i, data in enumerate(abide_loader):
        data = data.to(args.device)
        Sag = SAGNet(args).to(args.device)
        downsample += Sag(data).cpu().detach().numpy().tolist()
        label += data.y.cpu().detach().numpy().tolist()
        del Sag
        torch.cuda.empty_cache()

    downsample_df = pd.DataFrame(downsample)
    downsample_df['label'] = label

    # 原始保存路径
    downsample_dir = os.path.join(args.data_dir, 'Graph_pooling')  # 设定存储目录
    if not os.path.exists(downsample_dir):
        os.makedirs(downsample_dir)
    downsample_file = os.path.join(downsample_dir, 'data_pool_%.3f_.txt' % args.pooling_ratio)  # 设定文件名
    downsample_df.to_csv(downsample_file, index=False, header=False, sep='\t')  # 存储到文件

    # 创建Features文件夹及每个fold的子文件夹
    features_dir = os.path.join(args.data_dir, 'Features')
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    # 对于fold_1至fold_10，保存文件为features.txt
    for fold_num in range(1, 11):
        fold_dir = os.path.join(features_dir, f'fold_{fold_num}')
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)

        # 构造文件路径
        feature_file = os.path.join(fold_dir, 'features.txt')

        # 保存数据
        downsample_df.to_csv(feature_file, index=False, header=False, sep='\t')

    # 清理资源
    del data
    del abide_dataset
    del abide_loader
    del downsample_df
    torch.cuda.empty_cache()  # 清理CUDA缓存
def test_gcn(loader, model, args, test=True):
    """
    在加载器上测试GCN的性能。在GCN中，我们没有使用验证集。
    因此，这个函数用于打印测试集上的性能
    :param loader: torch_geometric.data.Dataloader的一个实例
    :param model: GCN模型的一个实例
    :param args: 从main.py传递的参数
    :param test: 布尔值，指示是否在测试集上进行评估（默认为True），如果为False，则在验证集上进行评估
    :return: 测试集上的准确率，损失，预测结果
    """
    model.eval()  # 将模型设置为评估模式
    correct = 0.0  # 初始化预测正确的样本数
    loss_test = 0.0  # 初始化测试损失
    output = []  # 初始化用于存储输出的列表
    criterion = nn.BCEWithLogitsLoss()  # 定义损失函数，这里使用带logits的二元交叉熵损失

    for data in loader:
        data = data.to(args.device)  # 将数据移至指定设备
        # 前向传播，这里假设model的输入为节点特征x，边索引edge_index和边属性edge_attr
        out, _ = model(data.x, data.edge_index, data.edge_attr)
        # 将输出从GPU（如果有的话）移至CPU，并转换为numpy数组后添加到output列表
        output += out.cpu().detach().numpy().tolist()

        if test:
            # 如果在测试集上进行评估
            pred = (out[data.test_mask] > 0.5).long()  # 使用阈值0进行二分类预测
            length = data.test_mask.sum().item()  # 测试集样本数
            correct += pred.eq(data.y[data.test_mask]).sum().item()  # 计算预测正确的样本数
            loss_test += criterion(out[data.test_mask], data.y[data.test_mask].float()).item()  # 计算测试损失
        else:
            # 如果不在测试集上进行评估（即在验证集上）
            pred = (out[data.val_mask] > 0.5).long()  # 类似地，进行预测
            length = data.val_mask.sum().item()  # 验证集样本数
            correct += pred.eq(data.y[data.val_mask]).sum().item()  # 计算预测正确的样本数
            loss_test += criterion(out[data.val_mask], data.y[data.val_mask].float()).item()  # 计算验证损失

    # 返回测试集（或验证集）上的准确率，损失，以及所有输出的列表
    return correct / length, loss_test / length, output  # 注意：这里假设返回平均损失，因此除以了样本数length

def train_gcn(dataloader, model, optimizer, save_path, args):
    """
    GCN的训练阶段。此处不使用验证集。
    :param dataloader: 训练集的数据加载器
    :param model: GCN模型的一个实例
    :param optimizer: 优化器，默认为Adam
    :param save_path: 当前进度的工作路径
    :param args: 从main.py传递的参数
    :return: 最佳模型的文件名
    """
    min_loss = 1e10  # 初始化最小损失为一个较大的数
    patience_cnt = 0  # 初始化耐心计数器，用于早停
    loss_set = []  # 损失记录列表
    acc_set = []  # 准确率记录列表
    best_epoch = 0  # 最佳轮次初始化
    num_epoch = 0  # 当前轮次计数器

    t = time.time()  # 记录开始时间

    for epoch in range(args.epochs):  # 遍历指定的训练轮次
        model.train()  # 将模型设置为训练模式
        loss_train = 0.0  # 初始化训练损失
        correct = 0  # 初始化预测正确的样本数
        num_epoch += 1  # 轮次计数器加1
        for i, data in enumerate(dataloader):  # 遍历数据加载器中的每个批次
            optimizer.zero_grad()  # 清除梯度
            data = data.to(args.device)  # 将数据移至指定设备
            out, _ = model(data.x, data.edge_index, data.edge_attr)  # 前向传播
            criterion = nn.BCEWithLogitsLoss()  # 定义损失函数
            loss = criterion(out[data.train_mask], data.y[data.train_mask].float())  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            loss_train += loss.item()  # 累加训练损失
            pred = (out[data.train_mask] > 0.5).long()  # 预测
            correct += pred.eq(data.y[data.train_mask]).sum().item()  # 计算预测正确的样本数

        # 计算训练集上的准确率
        acc_train = correct / data.train_mask.sum().item()

        # 在验证集上测试GCN模型，返回验证集上的准确率、损失和其他可能的信息（此处未使用）
        acc_val, loss_val, _ = test_gcn(dataloader, model, args, test=False)

        # 如果设置了详细模式，则打印当前轮次的信息
        if args.verbose:
            print('\r', 'Epoch: {:06d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
                  # 打印当前轮次、训练损失和训练准确率
                  'acc_train: {:.6f}'.format(acc_train), 'loss_val: {:.6f}'.format(loss_val),  # 打印验证损失和验证准确率
                  'acc_val: {:.6f}'.format(acc_val), 'time: {:.6f}'.format(time.time() - t),  # 打印耗时
                  flush=True, end='')  # 刷新输出流并避免换行

        # 将验证损失和验证准确率添加到对应的列表中
        loss_set.append(loss_val)
        acc_set.append(acc_val)

        # 如果当前轮次小于最小轮次限制，则继续下一轮
        if epoch < args.least:
            continue

            # 如果当前验证损失小于最小损失，则保存模型并更新相关变量
        if loss_set[-1] < min_loss:
            model_state = {'net': model.state_dict(), 'args': args}  # 打包模型状态和参数
            torch.save(model_state, os.path.join(save_path, '{}.pth'.format(epoch)))  # 保存模型
            min_loss = loss_set[-1]  # 更新最小损失
            best_epoch = epoch  # 更新最佳轮次
            patience_cnt = 0  # 重置耐心计数器
        else:
            patience_cnt += 1  # 增加耐心计数器

        # 如果耐心计数器达到上限，则提前结束训练
        if patience_cnt == args.patience:
            break

            # 清理保存路径下非最佳模型的文件
        files = [f for f in os.listdir(save_path) if f.endswith('.pth')]  # 获取所有.pth文件
        for f in files:
            if f.startswith('fold'):  # 跳过以'fold'开头的文件
                continue
            epoch_nb = int(f.split('.')[0])  # 提取文件名中的轮次号
            if epoch_nb != best_epoch:  # 如果不是最佳轮次对应的模型文件
                os.remove(os.path.join(save_path, f))  # 删除该文件

    # 如果设置了详细模式，则打印优化完成的信息以及总耗时
    if args.verbose:
        print('\n优化完成！总耗时: {:.6f}秒'.format(time.time() - t))  # 打印优化完成的信息和从开始到结束的总时间

    # 返回最佳轮次
    return best_epoch  # 返回在整个训练过程中达到最小验证损失的轮次
