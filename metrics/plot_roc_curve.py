import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import  roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_roc_curve(y_true, y_probs, dataset_type="Test"):
    """
    绘制ROC曲线并打印AUC值
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20
    # 定义类别名称
    # class_labels = ["LN0", "LN1-3", "LN4+"]
    class_labels = ["HER2-low", "HER2-zero", "HER2-positive"]
    # 将y_true转换为二进制标签矩阵
    y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))
    n_classes = y_true_binarized.shape[1]

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算宏平均 ROC AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"], roc_auc["macro"] = all_fpr, mean_tpr, auc(all_fpr, mean_tpr)

    # 计算微平均 ROC AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # 打印AUC值
    print(f"{dataset_type} ROC AUC values:")
    for i in range(n_classes):
        print(f"Class {class_labels[i]} AUC: {roc_auc[i]:.2f}")
    print(f"Macro Average AUC: {roc_auc['macro']:.2f}")
    print(f"Micro Average AUC: {roc_auc['micro']:.2f}")

    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))

    colors = ['#9edae5', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{class_labels[i]} ROC curve (AUC = {roc_auc[i]:.2f})')


    plt.plot(fpr["macro"], tpr["macro"], color='#000080', linestyle='-.', lw=4, label=f'Macro-average ROC curve (AUC = {roc_auc["macro"]:.2f})')
    plt.plot(fpr["micro"], tpr["micro"], color='#800080', linestyle='--', lw=4, label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset_type} ROC Curves')
    plt.legend(loc='lower right')
    plt.show()

    # 返回计算的 AUC 值字典
    return roc_auc

