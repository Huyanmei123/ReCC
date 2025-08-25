import os
import pickle as pkl
import sys
from torch_geometric import utils
import torch
import numpy as np
from config import *
import networkx as nx




def load_data(dataset_str,num_fea):
    if args.type == 'Binary':
        names = [args.feature_type,  'graph', 'train_mask', 'test_mask', 'labeled','regular_simi']
    else:
        names = [args.feature_type, 'graph', 'SIR']
    objects = []
    for i in range(len(names)):
        with open("../data/{}/{}-{}.{}".format(dataset_str, dataset_str, names[i], names[i]),
                  'rb') as f:
            if sys.version_info > (3, 0):

                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    if args.type == 'Binary':
        feature, graph, train_mask, test_mask, label, regular_simi = tuple(objects)
    else:
        feature, graph, label = tuple(objects)

    data = utils.from_networkx(graph)
    # test_feature = np.ones((graph.number_of_nodes(), 4))
    # print(test_feature)
    if args.type != 'Binary':
        node_id = np.arange(data.num_nodes)
        train_id = node_id[:int(data.num_nodes * 0.7)]
        test_id = node_id[int(data.num_nodes * 0.7):]
    data.x = torch.tensor(np.array(feature), dtype=torch.float32)[:,:num_fea]
    if args.type == 'Binary':
        label = torch.tensor(np.array(label), dtype=torch.long)
    else:
        label = torch.tensor(np.array(label), dtype=torch.float32)
    if args.CGCN:
        CGCN_feature = torch.tensor(np.array(CGCN_feature), dtype=torch.float32)
        CGCN_feature = torch.reshape(CGCN_feature, (-1, 1, 17, 17))

    #data.CGCN_x = CGCN_feature
    data.train_mask = train_mask
    data.test_mask = test_mask
    data.label = label
    data.regular_simi = regular_simi

    degrees = utils.degree(data.edge_index[0], data.num_nodes)
    data.nodes = list(graph.nodes())
    data.added_fea = degrees
    # data.added_fea = cal_centrality(dataset_str)


    print(data)
    return data


# Kendall algorithm
def kendall(a, b):
    Lens = len(a)
    ties_onlyin_x = 0
    ties_onlyin_y = 0
    con_pair = 0
    dis_pair = 0
    for i in range(Lens - 1):
        for j in range(i + 1, Lens):
            test_tying_x = np.sign(a[i] - a[j])
            test_tying_y = np.sign(b[i] - b[j])
            panduan = test_tying_x * test_tying_y
            if panduan == 1:
                con_pair += 1
            elif panduan == -1:
                dis_pair += 1
    Kendallta1 = (2 * (con_pair - dis_pair)) / (len(a) * (len(a) - 1))
    return Kendallta1


def get_rank(data):
    ranked_data = sorted(data)[::-1]
    rank = []
    for o_num in data:
        for r_num in ranked_data:
            if o_num == r_num:
                rank.append(ranked_data.index(o_num) + 1)
                break
    return rank

def cal_centrality(dataset_str):
    # names = ["ec_c++.txt",  'pc_c++.txt', 'Degree.txt', 'Extdegree.txt', 'Egodensity.txt','Conductance.txt','LCC.txt','Coredpearson1.txt']
    names = ['Degree.txt', 'Extdegree.txt', 'Egodensity.txt', 'Conductance.txt', 'LCC.txt',
             'Coredpearson1.txt']
    objects = []
    base_path = r"E:\Documents\代码\contragcn\GNNS\centrrality"

    for name in names:
        file_path = os.path.join(base_path, dataset_str, name)
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                value = float(parts[1]) if len(parts) >= 2 else float(parts[0])
                data.append(value)
        objects.append(data)
    res = torch.tensor(objects, dtype=torch.float32).T
    return res
