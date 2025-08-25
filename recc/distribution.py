import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from config import *
from utils import *
from CL.train import select_samples_based_on_Reg

num_fea = -1
if_flag = 1
abs_flag = 0

data = load_data(args.dataset, num_fea)

def distribution_control(if_flag,abs_flag):

    if abs_flag == 1:
        features,_ = torch.max(torch.abs(data.x), dim=1)
        abs_id = 'abs_'
    else:
        features,_ = torch.max(data.x, dim=1)
        abs_id = ''

    if if_flag == 1:
        if_indices = np.where(data.label == 1)
        if_features = features[if_indices]
        if_id = 'Influence'
    else:

        if_indices = np.where(data.label == 0)
        if_features = features[if_indices]
        if_id = 'Noninfluence'
    return if_features.numpy(),if_id,abs_id

def draw(features,if_id,abs_id):
    plt.figure(figsize=(12, 8))

    # 绘制直方图
    plt.subplot(3, 1, 1)
    plt.hist(features, bins=10, edgecolor='k')
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # 绘制箱线图
    plt.subplot(3, 1, 2)
    sns.boxplot(x=features)
    plt.title('Box Plot')
    plt.xlabel('Value')

    # 绘制核密度估计图
    plt.subplot(3, 1, 3)
    sns.kdeplot(features, shade=True)
    plt.title('Kernel Density Estimation')
    plt.xlabel('Value')
    plt.ylabel('Density')

    # 调整子图间距
    plt.suptitle(f'Distribution of {if_id} {args.dataset} {abs_id}sum_{data.x.shape[1]}_features', fontsize=16)
    plt.tight_layout()
    plt.show()

def similarity_check():
    influence_indices = torch.tensor(np.where(data.label == 1) or np.where(data.label == 2)).view(-1)
    noninfluence_indices = torch.tensor(np.where(data.label == 0) or np.where(data.label == 3) ).view(-1)
    pos,neg = select_samples_based_on_Reg()
    influence_pos = []
    influence_neg = []
    noninfluence_pos = []
    noninfluence_neg = []
    for i in influence_indices:
        influence_pos.extend(pos[i.item()])
        influence_neg.extend(neg[i.item()])
    for i in noninfluence_indices:
        noninfluence_pos.extend(pos[i.item()])
        noninfluence_neg.extend(neg[i.item()])

    influence_pos =  torch.tensor(influence_pos,dtype=int)
    influence_neg =  torch.tensor(influence_neg,dtype=int)
    noninfluence_pos =  torch.tensor(noninfluence_pos,dtype=int)
    noninfluence_neg =  torch.tensor(noninfluence_neg,dtype=int)

    influence_pos_counts = {element.item(): (influence_pos == element).sum().item() for element in influence_indices}
    in_pos_total_count = sum(influence_pos_counts.values())
    print(influence_pos_counts)
    print(f'Accuracy of influential nodes positive set selection: {in_pos_total_count / (influence_pos.shape[0] *2)}' )

    influence_neg_counts = {element.item(): (influence_neg == element).sum().item() for element in noninfluence_indices}
    in_neg_total_count = sum(influence_neg_counts.values())
    print(f'Accuracy of influential nodes negative set selection: {in_neg_total_count / (influence_neg.shape[0] *2)}' )

    nonifluence_pos_counts = {element.item(): (noninfluence_pos == element).sum().item() for element in noninfluence_indices}
    nonif_pos_total_counts = sum(nonifluence_pos_counts.values())
    print(f'Accuracy of noninfluential nodes positive set selection: {nonif_pos_total_counts / (noninfluence_pos.shape[0] *2)}' )

    nonifluence_neg_counts = {element.item(): (noninfluence_neg == element).sum().item() for element in influence_indices}
    nonif_neg_total_counts = sum(nonifluence_neg_counts.values())
    print(f'Accuracy of noninfluential nodes negative set selection: {nonif_neg_total_counts / (noninfluence_neg.shape[0] *2)}' )


# if_features,if_id,abs_id = distribution_control(if_flag=if_flag,abs_flag=abs_flag)
# draw(features=if_features,if_id=if_id,abs_id=abs_id)
similarity_check()

