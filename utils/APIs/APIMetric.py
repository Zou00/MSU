from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def api_metric(true_labels, pred_labels):
    """
    计算详细的评估指标
    """
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    
    # 计算详细的分类报告
    report = classification_report(true_labels, pred_labels, output_dict=True)
    
    # 添加混淆矩阵到报告中
    report['confusion_matrix'] = cm.tolist()
    
    return report