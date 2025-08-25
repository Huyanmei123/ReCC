# from CL.train import *
import numpy as np
import pandas as pd
from utils import *
import networkx as nx
import os

# def compute_cosine_matrix(features):
#     dim = features.shape[0]
#     cosine_matrix = torch.zeros(dim, dim)
#     for i in range(features.shape[0]):
#         vec_i = features[i].unsqueeze(0).expand(features.shape[0], -1)
#         cosine_similarities = cosine_similarity(features, vec_i, dim=1)
#         cosine_matrix[i] = cosine_similarities
#
#     # cosine_matrix.fill_diagonal_(-float('inf')) #除开自身
#     _, topK_indices = cosine_matrix.topk(2, dim=1)
#     _, bottomK_indices = cosine_matrix.topk(2, dim=1, largest=False)
#     return cosine_matrix, topK_indices, bottomK_indices
# def compute_euclidean(features):
#     features = features.cpu().numpy()
#     dim = features.shape[0]
#     euclidean_matrix = torch.zeros(dim, dim)
#     for i in range(dim):
#         for j in range(dim):
#             euclidean_matrix[i][j] = euclidean(features[i], features[j])
#
#     euclidean_matrix.fill_diagonal_(-float('inf')) #除开自身
#     _, topK_indices = euclidean_matrix.topk(2, dim=1, largest=False)
#     _, bottomK_indices = euclidean_matrix.topk(1000, dim=1)
#     return euclidean_matrix, topK_indices, bottomK_indices
# def clustering_loss(graph_emb,centriods,indices_vectors):
#     centers = torch.stack([centriods[i] for i in indices_vectors]).to(device)
#     emd = graph_emb.data
#     return torch.norm(emd[data.train_mask] - centers[data.train_mask], p=2) * 0.1 / 2
# def kmeans_clustering(data, centroids):
#     """
#     K-means 聚类算法，根据给定的质心 centroids 将数据 data 中的每个点分配到最近的簇。
#
#     Parameters:
#     - data: 数据集，每行代表一个数据点
#     - centroids: 初始质心列表，每行代表一个质心
#
#     Returns:
#     - labels: 每个数据点所属的簇的标签
#     """
#     num_data = data.shape[0]  # 数据点个数
#     num_clusters = centroids.shape[0]  # 簇的个数
#
#     # 初始化 labels 数组，用于存储每个数据点所属的簇的标签
#     labels = torch.zeros(num_data)
#
#     # 对数据中的每个点进行分配到最近质心的操作
#     for i in range(num_data):
#         # 计算数据点到所有质心的距离
#         distances = torch.linalg.norm(data[i] - centroids, axis=1)
#
#         # 找到距离最近的质心的索引，作为该数据点的簇标签
#         cluster_label = torch.argmin(distances)
#         labels[i] = cluster_label
#
#     return labels.long()
# def compute_jaccard_matrix(features):
#     features = features.cpu().numpy()
#     dim = features.shape[0]
#     jaccard_matrix = torch.zeros(dim, dim)
#     for i in range(dim):
#         for j in range(dim):
#             if i == j:
#                 jaccard_matrix[i][j] = 1.0  # 自身相似度为 1
#             else:
#                 jaccard_matrix[i][j] = jaccard_score(features[i], features[j], average='binary')
#
#     _, topK_indices = jaccard_matrix.topk(2, dim=1)
#     _, bottomK_indices = jaccard_matrix.topk(2, dim=1, largest=False)
#     return jaccard_matrix, topK_indices, bottomK_indices
# def compute_chebyshev_matrix(features):
#     features = features.cpu().numpy()
#     dim = features.shape[0]
#     chebyshev_matrix = cdist(features, features, metric='chebyshev')
#
#     chebyshev_matrix = torch.tensor(chebyshev_matrix)
#     chebyshev_matrix.fill_diagonal_(-float('inf')) #除开自身
#     _, topK_indices = chebyshev_matrix.topk(2, dim=1, largest=False)
#     _, bottomK_indices = chebyshev_matrix.topk(2, dim=1)
#     return chebyshev_matrix, topK_indices, bottomK_indices
# def compute_abslute_difference(feautres):
#     f = feautres
#     abs_sum_vector = torch.sum(torch.abs(f), dim=1, keepdim=True)
#
#     diff = abs_sum_vector - abs_sum_vector.T
#     distance_matrix = torch.sqrt(diff ** 2)
#
#     distance_matrix.fill_diagonal_(-float('inf')) #除开自身
#     _, topK_indices = distance_matrix.topk(2, dim=1, largest=False)
#     _, bottomK_indices = distance_matrix.topk(2, dim=1)
#
#     return distance_matrix, topK_indices, bottomK_indices

def harmonic_centrality(G):
    harmonics = {}
    n = len(G.nodes())

    for node in G.nodes():
        sum_inverse_distance = 0.0
        for target in G.nodes():
            if node != target:
                try:
                    shortest_path_length = nx.shortest_path_length(G, source=node, target=target)
                    if shortest_path_length > 0:
                        sum_inverse_distance += 1 / shortest_path_length
                except nx.NetworkXNoPath:
                    continue

        if sum_inverse_distance > 0:
            harmonics[node] = (n - 1) / sum_inverse_distance
        else:
            harmonics[node] = 0.0

    return harmonics
def dict_sort(dic):
    return dict(
                sorted(dic, key=lambda item: int(item[0]), reverse=False)
           )
def skeleton_centrality(G):
    # 计算节点的度
    degrees = dict(G.degree())

    # 计算图的平均度
    avg_degree = sum(degrees.values()) / len(G)

    # 确定骨干节点集合
    skeleton_nodes = {node for node, degree in degrees.items() if degree >= (avg_degree * 4)}

    # 构建骨干图
    skeleton_graph = G.subgraph(skeleton_nodes)

    # 计算骨干节点的度中心性
    skeleton_centrality = nx.degree_centrality(skeleton_graph)

    # 将不在骨干图中的节点的中心性设为0
    for node in G.nodes():
        if node not in skeleton_centrality:
            skeleton_centrality[node] = 0.0

    return skeleton_centrality

root_path = 'E:\Documents\Python Scripts\DataSet'
graph_target = '-graph.txt'
SIR_target = '-graph_SIR.txt'
graph_list = []
SIR_list = []
# 遍历根目录下的所有子目录
for root, dirs, files in os.walk(root_path):
    # 遍历当前子目录中的所有文件
    for file in files:
        # 构建文件的完整路径
        file_path = os.path.join(root, file)
        # 判断文件是否含有指定字符
        if graph_target in file:
            # 如果含有指定字符，就打印文件路径
            graph_list.append(file_path)
        if SIR_target in file:
            SIR_list.append(file_path)

# graph_list = [item for item in graph_list if 'ca-AstroPh' in item]
print(graph_list)
"""
Degree
Betweenness
Eigenvetor
Closenness
K-Core
Coreness
Load 
Information 
Proximity Prestige
Closeness Vitality
Skeleton
Social 
k-path
Core Structure
Core Centrality
"""

centrality_list = [
# "Degree",
# "Closenness",
# "Betweenness",
# "Eigenvetor",
# "PageRank",
# "harmonic",
# "Coreness",
# "Load",
# "Information",
# "Closeness Vitality",
"Skeleton",
]

for G_path in graph_list:
    name = G_path.split('\\')[-1].split('-graph')[0]
    print(name)
    G = nx.read_edgelist(G_path)
    SIR_path = [path for path in SIR_list if name in path]
    SIR = np.array(pd.read_csv(SIR_path[0],sep=' ',header=None,index_col=False))
    for i in centrality_list:
        print(i)
        if i == 'Degree':
            degree_centrality = nx.degree_centrality(G)
            sorted_degree_centrality = dict_sort(degree_centrality.items())
            centrality = np.array(list(sorted_degree_centrality.values())).reshape(-1, 1)
            SIR = np.concatenate((SIR, centrality), axis=1)
        elif i == 'Closenness':
            closeness_centrality = nx.closeness_centrality(G)
            sorted_closeness_centrality = dict_sort(closeness_centrality.items())
            centrality = np.array(list(sorted_closeness_centrality.values())).reshape(-1, 1)
            SIR = np.concatenate((SIR, centrality), axis=1)
        elif i == 'Betweenness':
            betweenness_centrality = nx.betweenness_centrality(G)
            sorted_betweenness_centrality = dict_sort(betweenness_centrality.items())
            centrality = np.array(list(sorted_betweenness_centrality.values())).reshape(-1, 1)
            SIR = np.concatenate((SIR, centrality), axis=1)
        elif i == 'Eigenvetor':
            eigenvector_centrality = nx.eigenvector_centrality(G)
            sorted_eigenvector_centrality = dict_sort(eigenvector_centrality.items())
            centrality = np.array(list(sorted_eigenvector_centrality.values())).reshape(-1, 1)
            SIR = np.concatenate((SIR, centrality), axis=1)
        elif i == 'PageRank':
            PageRank_centrality = nx.pagerank(G)
            sorted_PageRank_centrality = dict_sort(PageRank_centrality.items())
            centrality = np.array(list(sorted_PageRank_centrality.values())).reshape(-1, 1)
            SIR = np.concatenate((SIR, centrality), axis=1)
        elif i == "harmonic":
            harmonic = harmonic_centrality(G)
            print('centrality complete')
            sorted_harmonic = dict_sort(harmonic.items())
            centrality = np.array(list(sorted_harmonic.values())).reshape(-1, 1)
            SIR = np.concatenate((SIR, centrality), axis=1)
        elif i == "Coreness":
            coreness = nx.core_number(G)
            sorted_coreness = dict_sort(coreness.items())
            centrality = np.array(list(sorted_coreness.values())).reshape(-1, 1)
            SIR = np.concatenate((SIR, centrality), axis=1)
        elif i == 'Load':
            load = nx.load_centrality(G)
            sorted_load = dict_sort(load.items())
            centrality = np.array(list(sorted_load.values())).reshape(-1, 1)
            SIR = np.concatenate((SIR, centrality), axis=1)
        elif i == "Information":
            information = nx.information_centrality(G)
            sorted_information = dict_sort(information.items())
            centrality = np.array(list(sorted_information.values())).reshape(-1, 1)
            SIR = np.concatenate((SIR, centrality), axis=1)
        elif i == "Closeness Vitality":
            vitality = nx.closeness_vitality(G)
            sorted_vitality = dict_sort(vitality.items())
            centrality = np.array(list(sorted_vitality.values())).reshape(-1, 1)
            SIR = np.concatenate((SIR, centrality), axis=1)
        elif i == "Skeleton":
            skeleton = skeleton_centrality(G)
            sorted_skeleton = dict_sort(skeleton.items())
            centrality = np.array(list(sorted_skeleton.values())).reshape(-1, 1)
            SIR = np.concatenate((SIR, centrality), axis=1)
    columns = ['node','SIR']
    columns.extend(centrality_list)
    df = pd.DataFrame(SIR,columns=columns)
    df.to_csv(f'{root_path}\\{name}-skeleton.csv',index=False)
