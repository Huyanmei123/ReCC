import pickle as pkl
import pandas as pd
import numpy as np
import os

from config import *
from utils import *

path = 'E:\Documents\Python Scripts\DataSet\\'
out_path = r'E:\Documents\GNNS\GNNS\data\\'
def find_files_with_string(directory, search_string):
    matching_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if search_string in file:
                matching_files.append(os.path.join(root, file))
    return matching_files

SIR_path_list = find_files_with_string(path,'SIR.txt')
for item in SIR_path_list:
    data_name = item.split('\\')[-1].split('-graph')[0]
    if data_name == 'lastfm_asia':
        data = np.array(pd.read_csv(item,sep=' ',index_col=False,header=None))
        print(data_name)

        # 获取value列
        values = data[:, -1]

        # 按降序排序值并获取排序后的索引
        sorted_indices = np.argsort(-values)

        # 获取每个百分比段的分界索引
        n = len(values)
        percentile_1 = int(n * 0.08)
        # percentile_2 = int(n * 0.05)
        # percentile_3 = int(n * 0.2)

        # 获取不同百分比段的索引
        top_Fir_percent_idx = sorted_indices[:percentile_1]
        # top_Sec_percent_idx = sorted_indices[percentile_1:percentile_2]
        # top_Thr_percent_idx = sorted_indices[percentile_2:percentile_3]
        #
        labels = np.zeros(data.shape[0])
        labels[top_Fir_percent_idx] = 1
        # labels[top_Sec_percent_idx] = 2
        # labels[top_Thr_percent_idx] = 3

        cluster_name = '8_re'
        with open(f'{out_path}{data_name}\\{data_name}-{cluster_name}.{cluster_name}', 'wb') as f:
            pkl.dump(labels, f)
        print(f'{out_path}{data_name}\\{data_name}-{cluster_name}.{cluster_name}')


#打印结果
# print("Top 5% key indices:", data[top_5_percent_idx, 0])
# print("5%-15% key indices:", data[between_5_and_15_percent_idx, 0])
# print("15%-35% key indices:", data[between_15_and_35_percent_idx, 0])
# print("35%-100% key indices:", data[bottom_35_to_100_percent_idx, 0])




