import argparse  # 导入 argparse 库，用于解析命令行参数
from matplotlib.pyplot import cm  # 导入颜色映射功能
from numpy import interp  # 导入 NumPy 的插值函数
from sklearn.metrics import auc,roc_curve
from sklearn.model_selection import KFold
import matplotlib
from matplotlib import rcParams
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from models import MH_GCAT   # 导入你的预训练模型
import random
matplotlib.use('TkAgg')  # 或 'Qt5Agg'
# 设置中文字体和负号显示
rcParams['font.sans-serif'] = ['SimHei']  # 替换为系统支持的中文字体
rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

plt.rcParams.update({'font.size': 18})
def set_random_seeds(seed):
    """设置随机种子，确保结果可复现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)  # Python 内置的随机数种子
    torch.cuda.manual_seed_all(seed)  # GPU 计算的随机种子



def draw_cv_roc_curve(cv, out, y, thre=0, title=''):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)  # Define standardized FPR (False Positive Rate)
    plt.figure(figsize=(10, 7))  # Set figure size

    i = 0  # Counter for cross-validation folds
    for train, test in cv.split(out, y):  # Iterate over CV train-test splits
        probas_ = out.iloc[test]  # Get predicted probabilities for the test set
        preds = [int(item) for item in (probas_.iloc[:, i].values > thre)]  # Generate predictions based on threshold
        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_.iloc[:, i])
        tprs.append(interp(mean_fpr, fpr, tpr))  # Interpolate TPR over standardized FPR
        tprs[-1][0] = 0.0  # Set initial TPR to 0
        roc_auc = auc(fpr, tpr)  # Calculate AUC
        aucs.append(roc_auc)  # Save AUC value
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %.4f)' % (i + 1, roc_auc))  # Plot ROC for each fold

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Random prediction', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # Set final TPR to 1
    mean_auc = auc(mean_fpr, mean_tpr)  # Calculate mean AUC
    std_auc = np.std(aucs)  # Calculate standard deviation of AUC
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %.4f $\pm$ %.4f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)  # Plot mean ROC curve
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 Standard Deviation')

    # Set plot style
    plt.xlim([-0.05, 1.05])  # Set x-axis range
    plt.ylim([-0.05, 1.05])  # Set y-axis range
    plt.xlabel('False Positive Rate')  # Set x-axis label
    plt.ylabel('True Positive Rate')  # Set y-axis label
    plt.title('ROC Curve')  # Set plot title
    plt.legend(loc="lower right")  # Set legend location
    plt.suptitle(title)  # Set overall title
    plt.show()  # Display the plot

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()

    # 添加命令行参数
    parser.add_argument('--roc', action='store_true', default=True, help='可视化 ROC 曲线和平均混淆矩阵')
    parser.add_argument('--embedding', action='store_true', default=False, help='可视化学习到的节点嵌入')
    parser.add_argument('--group', type=str, default='gender', help='用于分组的表型属性')
    parser.add_argument('--result_root', type=str, default='./results', help='结果保存的根目录')
    parser.add_argument('--model_root', type=str, default='./checkpoints', help='存储模型的根目录')
    parser.add_argument('--data_root', type=str, default='./data', help='数据的根目录')
    parser.add_argument('--seed', type=int, default=13, help='随机种子')
    parser.add_argument('--pooling_ratio', type=float, default=0.05, help='池化比例')
    # 解析命令行参数
    args = parser.parse_args()
    # 绘制 ROC 曲线和平均混淆矩阵
    print("开始")
    if args.roc:
        threshold = 0.5  # 设置阈值
        result_path = args.result_root  # 获取结果文件路径
        assert os.path.exists(result_path), \
            '未找到分类结果'  # 检查结果路径是否存在

        # 找到匹配的结果文件
        file_name = [
            f for f in os.listdir(result_path)
            if f.startswith('GCN_pool') and
               f.split('_')[3] == 'seed' and  # 确保文件名包含 'seed' 标志
               int(f.split('_')[4].split('.')[0]) == args.seed  # 提取 'seed' 后的整数部分
        ]

        assert len(file_name), \
            '未找到匹配要求的结果: ' \
            '池化比例 {:.3f}, 随机种子: {:d}'.format(args.pooling_ratio, args.seed)
        file_name = file_name[0]

        # 加载预测结果
        pred = pd.read_csv(os.path.join(result_path, file_name))

        # 初始化 K 折交叉验证
        kf = KFold(n_splits=10, random_state=args.seed, shuffle=True)

        # 加载真实标签
        labels = pd.read_csv(os.path.join(args.data_root, 'phenotypic', 'log.csv'))['label']
        print("# 绘制 ROC 曲线")
        # 绘制 ROC 曲线
        draw_cv_roc_curve(kf, pred, labels, thre=threshold,
                          title='pooling_ratio = {:.3f}, seed= {:d}'.
                          format(args.pooling_ratio, args.seed))

    # 可视化节点嵌入
    if args.embedding:
        print("#可视化节点嵌入")

        # 1. 加载预训练 GCN 模型
        check_path = os.path.join(args.model_root, 'GCN')
        models = [f for f in os.listdir(check_path) if f.startswith('fold') and f.endswith('.pth')]
        assert len(models), "未找到训练好的 GCN 模型。"

        # 选择性能最好的模型
        test_acc = [float(f.split('_')[3]) for f in models]
        best_index = np.argmax(test_acc)
        fold_num = int(models[best_index].split('_')[1])
        model_file = os.path.join(check_path, models[best_index])

        checkpoint = torch.load(model_file)
        model_args = checkpoint['args']
        gcn_model = MH_GCAT(model_args).to(model_args.device)
        gcn_model.load_state_dict(checkpoint['net'])
        gcn_model.eval()

        # 2. 加载特征和图数据
        features_path = os.path.join(args.data_root, 'Features',
                                     'fold_%d' % fold_num, 'features.txt')
        features = pd.read_csv(features_path, header=None, sep='\t')
        x = torch.tensor(features.iloc[:, :-1].values, dtype=torch.float)  # 节点特征

        adj_path = os.path.join(args.data_root, 'population graph', 'graphcos.adj')
        attr_path = os.path.join(args.data_root, 'population graph', 'graphcos.attr')
        adj_path2 = os.path.join(args.data_root, 'population graph', 'graphnoimg.adj')
        attr_path2 = os.path.join(args.data_root, 'population graph', 'graphnoimg.attr')

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

        edge_index = torch.tensor(new_edge_index, dtype=torch.long)
        edge_attr = torch.tensor(new_edge_attr, dtype=torch.float)

        # 创建 PyTorch Geometric 数据对象
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # 生成节点嵌入
        with torch.no_grad():
            data = data.to(model_args.device)
            _, embeddings = gcn_model(data.x, data.edge_index, data.edge_attr)
        embeddings = embeddings.cpu().numpy()

        # 加载分组信息
        logs = pd.read_csv(os.path.join(args.data_root, 'phenotypic', 'log.csv'))


        # 根据分组类别生成标签
        def generate_tags(logs, group):
            if group == 'gender':
                sex = ['Female', 'Male']
                return np.array([sex[2 - i] for i in logs['SEX'].values])
            elif group == 'site':
                return logs['SITE_ID'].values
            elif group == 'label':
                # 将0和1映射为 'TC' 和 'ASD'
                return np.array(['TC' if label == 0 else 'ASD' for label in logs['label'].values])
            elif group == 'age':
                sample_ages = []
                for age in logs['AGE_AT_SCAN'].values:
                    if age <= 12:
                        sample_ages.append('0 <= age <= 12')
                    elif age <= 17:
                        sample_ages.append('13 <= age <= 17')
                    else:
                        sample_ages.append('18 <= age <= 58')
                return np.array(sample_ages)
            else:
                raise AttributeError(f"没有可用的分组: {group}")


        tags = generate_tags(logs, args.group)

        # 打印每个标签的数量
        unique_tags, counts = np.unique(tags, return_counts=True)
        print("标签分布:")
        for tag, count in zip(unique_tags, counts):
            print(f"{tag}: {count}")

        # 固定的颜色映射，确保一致性
        fixed_colors = {
            'TC': '#FF0000',  # '0' -> TC
            'ASD': '#008080',  # '1' -> ASD
            'Female': '#FF9900',
            'Male': '#0000FF',
            '0 <= age <= 12': '#E1AFAB',
            '13 <= age <= 17': '#FFD964',
            '18 <= age <= 58': '#B4C6E7',
        }

        unique_tags = np.unique(tags)
        if args.group not in ['label','gender', 'age']:
            # 为其他分组动态生成颜色映射
            fixed_colors = {tag: cm.Set3(i / len(unique_tags)) for i, tag in enumerate(unique_tags)}

        # 使用 t-SNE 对节点嵌入降维
        scaler = StandardScaler()  # 标准化特征
        normalized_embeddings = scaler.fit_transform(embeddings)

        tsne = TSNE(n_components=2, random_state=args.seed, perplexity=50, learning_rate=300, init='random')
        embeddings_2d = tsne.fit_transform(normalized_embeddings)

        # t-SNE 降维到二维 (原始特征)
        if torch.is_tensor(features):
            features = features.detach().cpu().numpy()

        features_2d = TSNE(n_components=2).fit_transform(features)

        # 绘制原始特征的二维散点图
        plt.figure(figsize=(7, 6))
        for tag in unique_tags:
            selected = tags == tag
            plt.scatter(features_2d[selected, 0], features_2d[selected, 1],
                        s=30, color=fixed_colors.get(tag, tag), label=str(tag))  # 使用固定颜色映射
        # 设置图例框大小和位置，并调整宽度
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.2),
            fontsize=16,  # 增大字母大小
            ncol=4,
            frameon=True,
            handlelength=10,  # 控制图例框架的宽度
            labelspacing=1  # 控制标签之间的间距
        )
        plt.title('Features', fontsize=20)  # 增大标题字体大小
        plt.tight_layout()  # 优化布局，确保图例在下面

        # 保存图像为高分辨率
        plt.savefig('features_plot.png', dpi=300)
        plt.show()

        # 绘制节点嵌入的二维散点图
        plt.figure(figsize=(7, 6))
        for tag in unique_tags:
            selected = tags == tag
            plt.scatter(embeddings_2d[selected, 0], embeddings_2d[selected, 1],
                        s=30, color=fixed_colors.get(tag, tag), label=str(tag), alpha=1)  # 使用固定颜色映射
        # 设置图例框大小和位置，并调整宽度
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.2),
            fontsize=16,  # 增大字母大小
            ncol=4,
            frameon=True,
            handlelength=10,  # 控制图例框架的宽度
            labelspacing=2  # 控制标签之间的间距
        )
        plt.title('Node Embeddings', fontsize=20)  # 增大标题字体大小
        plt.tight_layout()  # 优化布局，确保图例在下面

        # 保存图像为高分辨率
        plt.savefig('node_embeddings_plot.png', dpi=300)
        plt.show()
