import random
import time

import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt
from torch import nn, optim
from GCNEmbedding import *
from GATembedding import *
from utils import *
from torch.nn.functional import cosine_similarity
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import adjusted_rand_score,accuracy_score,jaccard_score,normalized_mutual_info_score,f1_score
import sys

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

color_red = "\033[1;31m"
color_green = "\033[1;32m"
color_reset = "\033[0m"

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("cuda ", torch.cuda.is_available())







num_fea = 2
#
data = load_data(args.dataset, num_fea)
# data = load_data(sys.argv[1], num_fea)
# data.x = data.x[:,-1].reshape(-1,1)
data.to(device)

learing_data_indices = data.train_mask + data.test_mask
data_mask = torch.as_tensor(learing_data_indices == 1, device=device)  # [N]


def supervised_select_samples_based_on_edges():

    indices_dic = {}
    for label in range(cluster_number):
        this_label_indices = np.where(data.label.cpu().numpy() == label)[0]
        indices_dic[label] = torch.tensor(this_label_indices)

    distance_matrix = torch.tensor(data.regular_simi)


    topK_indices = {}
    bottomk_indices = {}


    inf_dic = {}
    ninf_dic = {}
    for this_label in list(indices_dic.keys()):
        other_labels = list(indices_dic.keys())
        if this_label in other_labels:
            other_labels.remove(this_label)
        for label in other_labels:
            inf_list = []
            inf_list.append(indices_dic[label])
            inf_dic[label] = set_sample_matrix(distance_matrix,inf_list,float('inf'))
            ninf_dic[label] = set_sample_matrix(distance_matrix, inf_list, float('-inf'))

    for i in range(distance_matrix.shape[0]):
        for label in list(indices_dic.keys()):
            if torch.eq(indices_dic[label],i).any().item():
                _,this_topk = inf_dic[label][i].topk(k=2, largest=True)
                if i not in topK_indices:
                    topK_indices[i] = []
                topK_indices[i].append(this_topk.tolist())
                other_labels = list(indices_dic.keys())
                for other_label in other_labels:
                    _,this_bottomK = ninf_dic[other_label][i].topk(k=2, largest=False)
                    # selected_this_bottomK = random.sample(this_bottomK.tolist(), 2)
                    if i not in bottomk_indices:
                        bottomk_indices[i] = []
                    bottomk_indices[i].extend(this_bottomK.tolist())
                break


    return topK_indices, bottomk_indices

def fix_missing_number(lst):
    n = len(lst)
    for i in range(1, n):

        if lst[i] - lst[i-1] != 1:

            for j in range(i, n):
                lst[j] -= 1
            break
    return lst

def select_samples_based_on_edges():
    degree = data.degree[:,1].view(1,-1)
    degree_sorted_indices = torch.argsort(degree,descending=True)
    top_indices = degree_sorted_indices[:int(degree_sorted_indices.shape[0]*0.01)]
    sec_indices = degree_sorted_indices[int(degree_sorted_indices.shape[0]*0.01):int(degree_sorted_indices.shape[0]*0.05)]
    thr_indices = degree_sorted_indices[int(degree_sorted_indices.shape[0]*0.05):int(degree_sorted_indices.shape[0]*0.2)]
    remain_indiecs = degree_sorted_indices[int(degree_sorted_indices.shape[0]*0.2):]

    diff = degree - degree.T
    distance_matrix = torch.sqrt(diff ** 2)

    pos_numbers = 2
    neg_numbers = 3

    topK_indices = torch.zeros((degree.shape[-1], pos_numbers))
    bottomk_indices = torch.zeros((degree.shape[-1], neg_numbers))

    top_inf = set_sample_matrix(distance_matrix,sec_indices,thr_indices,remain_indiecs,float('inf'))
    sec_inf = set_sample_matrix(distance_matrix,top_indices,thr_indices,remain_indiecs,float('inf'))
    thr_inf = set_sample_matrix(distance_matrix,top_indices,sec_indices,remain_indiecs,float('inf'))
    re_inf = set_sample_matrix(distance_matrix,top_indices,sec_indices,thr_indices,float('inf'))
    top_minf = set_sample_matrix(distance_matrix,sec_indices,thr_indices,remain_indiecs,float('-inf'))
    sec_minf = set_sample_matrix(distance_matrix,top_indices,thr_indices,remain_indiecs,float('-inf'))
    thr_minf = set_sample_matrix(distance_matrix,top_indices,sec_indices,remain_indiecs,float('-inf'))
    re_minf = set_sample_matrix(distance_matrix,top_indices,sec_indices,thr_indices,float('-inf'))

    for i in range(distance_matrix.shape[0]):
        if torch.eq(top_indices,i).any().item():
            _,topK_indices[i] = top_inf[i].topk(k = 2,largest = False)
            _,bottomk_indices[i,0] = sec_minf[i].topk(k = 1, largest = True)
            _,bottomk_indices[i,1] = thr_minf[i].topk(k = 1, largest = True)
            _,bottomk_indices[i,2] = re_minf[i].topk(k = 1, largest = True)
        elif torch.eq(sec_indices,i).any().item():
            _,topK_indices[i] = sec_inf[i].topk(k = 2,largest = False)
            _,bottomk_indices[i,0] = top_minf[i].topk(k = 1, largest = True)
            _,bottomk_indices[i,1] = thr_minf[i].topk(k = 1, largest = True)
            _,bottomk_indices[i,2] = re_minf[i].topk(k = 1, largest = True)
        elif torch.eq(thr_indices,i).any().item():
            _,topK_indices[i] = thr_inf[i].topk(k = 2,largest = False)
            _,bottomk_indices[i,0] = top_minf[i].topk(k = 1, largest = True)
            _,bottomk_indices[i,1] = sec_inf[i].topk(k = 1, largest = True)
            _,bottomk_indices[i,2] = re_minf[i].topk(k = 1, largest = True)
        elif torch.eq(remain_indiecs,i).any().item():
            _,topK_indices[i] = re_inf[i].topk(k = 2,largest = False)
            _,bottomk_indices[i,0] = top_minf[i].topk(k = 1, largest = True)
            _,bottomk_indices[i,1] = sec_minf[i].topk(k = 1, largest = True)
            _,bottomk_indices[i,2] = thr_minf[i].topk(k = 1, largest = True)
        else:
            print('Unexpected false!')
            break

    return topK_indices.long(),bottomk_indices.long()

def set_sample_matrix(distance,other_indices,type):
    diff = distance.clone()
    diff.fill_diagonal_(type)
    for other in other_indices:
        diff[:,other] = type

    return diff

def select_samples_based_on_Reg():

    regular_simi_matrix = data.regular_simi.clone().detach()
    print(regular_simi_matrix)

    pos_numbers = 3
    neg_numbers = int(data.label.shape[0]*0.02)

    topK_dic = {}
    bottomk_dic = {}

    for i in range(regular_simi_matrix.shape[0]):
        if i not in topK_dic :
            topK_dic[i] = []
        if i not in bottomk_dic :
            bottomk_dic[i] = []
        _,topK_indices = regular_simi_matrix[i].topk(k=pos_numbers,largest=True)
        topK_indices = topK_indices.tolist()
        topK_indices.remove(i)
        _,bottomk_indices = regular_simi_matrix[i].topk(k=neg_numbers,largest=False)
        topK_dic[i].extend(topK_indices)
        # bottomk_dic[i].extend(random.sample(bottomk_indices.tolist(),math.ceil(neg_numbers*0.01)))
        bottomk_dic[i].extend(random.sample(bottomk_indices.tolist(), 1))

    return topK_dic,bottomk_dic

def compute_clustering_absolute_diff(feautres):
    f = feautres
    abs_sum_vector = torch.sum(torch.abs(f), dim=1, keepdim=True).T
    # abs_sum_vector = torch.max(torch.abs(f),dim=1)

    pos_sets_number = 2
    neg_sets_number = 2

    """Contrastive sets control"""
    if_indices = torch.empty(1)
    nif_indices = torch.empty(1)
    # num_top = int(abs_sum_vector.shape[1] * 0.05)
    _,t_indices = torch.sort(abs_sum_vector,descending=True)
    # if_indices = t_indices[0,:num_top]
    # nif_indices = t_indices[0,num_top:]
    if_mask = torch.zeros(abs_sum_vector.shape[1])
    # if_mask[if_indices] = 1

    diff = abs_sum_vector - abs_sum_vector.T
    distance_matrix = torch.sqrt(diff ** 2)
    distance_matrix.fill_diagonal_(float('-inf')) #除开自身

    up_diff_matrix = distance_matrix.clone()
    down_diff_matrix = distance_matrix.clone()
    # up_diff_matrix[:,nif_indices] = float('-inf')
    # down_diff_matrix[:,if_indices] = float('-inf')


    topK_indices = torch.zeros((distance_matrix.shape[0],pos_sets_number))
    bottomK_indices = torch.zeros((distance_matrix.shape[0],neg_sets_number))

    for i in range(distance_matrix.shape[0]):
        if torch.eq(if_indices, i).any().item():
            _, bottomK_indices[i] = down_diff_matrix[i].topk(k=neg_sets_number,largest=True)
            _, topK_indices[i] = up_diff_matrix[i].topk(k=pos_sets_number,largest=False)
        elif torch.eq(nif_indices, i).any().item():
            _, bottomK_indices[i] = up_diff_matrix[i].topk(k=neg_sets_number,largest=True)
            _,  topK_indices[i]= down_diff_matrix[i].topk(k=pos_sets_number,largest=False)
        else:
            _, bottomK_indices[i] = distance_matrix[i].topk(k=neg_sets_number,largest=True)
            _,  topK_indices[i]= distance_matrix[i].topk(k=pos_sets_number,largest=False)
    topK_indices = topK_indices.long()
    bottomK_indices = bottomK_indices.long()
    return topK_indices,bottomK_indices

def clustering_accuracy(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    accuracy = sum(w[row_ind[i], col_ind[i]] for i in range(len(row_ind))) / y_pred.size
    return accuracy




def infoNCE_loss(x,topk,bottomk,temperature=0.1):

    N, K1 = topk.shape
    # _, K2 = bottomk.shape

    # x = graph_emb  # [N, D]

    pos = x[topk]  # [N, K1, D]
    neg = x[bottomk]  # [N, K2, D]

    q = x.unsqueeze(1)  # [N, 1, D]
    pos_sim = F.cosine_similarity(q, pos, dim=-1)  # [N, K1]
    neg_sim = F.cosine_similarity(q, neg, dim=-1)  # [N, K2

    logits_pos = pos_sim / temperature  # [N, K1]
    logits_neg = neg_sim / temperature  # [N, K2]

    pos_lse = torch.logsumexp(logits_pos, dim=1)  # [N]
    neg_lse = torch.logsumexp(logits_neg, dim=1)  # [N]

    all_lse = torch.logaddexp(pos_lse, neg_lse)  # [N]

    loss_per_node = -(pos_lse - all_lse)  # [N]
    loss = torch.sum(loss_per_node * data_mask) / N

    return loss

def KL_loss(graph_emb,centers):
    distances = torch.zeros((graph_emb.shape[0], centers.shape[0]))
    for i in range(graph_emb.shape[0]):
        for j in range(centers.shape[0]):
            distances[i,j] = torch.norm(graph_emb[i]-centers[j],p=2)
    distances = 1 / (1 + distances)
    distances_sum = torch.sum(distances,dim=1,keepdim=True)
    q_ij_matrix = distances / distances_sum
    f_j = distances.sum(dim = 0)
    p_ij_matrix = (q_ij_matrix ** 2 / f_j) / torch.sum((q_ij_matrix ** 2 / f_j ), dim=1,keepdim=True)
    kl_loss = torch.sum(p_ij_matrix * torch.log(p_ij_matrix / q_ij_matrix))
    return kl_loss,p_ij_matrix

def reconstruction_loss(A,embedding):
    A_hat = F.sigmoid(torch.matmul(embedding, embedding.T))
    r_loss = torch.norm(A - A_hat)
    return r_loss

def train_model(data,topk,bottomk):
    model.train()
    count_loss = float('inf')
    optimizer.zero_grad()
    out = model(data)
    loss = infoNCE_loss(out,topk,bottomk)
    loss.backward()
    optimizer.step()
    print(loss)
    if count_loss > loss:
        count_loss == loss
        torch.save(model.state_dict(), 'model_state_dict.pth')
    return out

def train_clustering_model(data,topk,bottomk,centers):
    model.train()
    count_loss = float('inf')
    out = model(data)
    # centers = model.centers
    # k_input = out.detach().cpu().numpy()

    loss = infoNCE_loss(out,topk,bottomk) + KL_loss(graph_emb=out[learing_data_indices],centers=centers)[0]

   
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    return loss,out

def train_no_kl_model(data, topk, bottomk):
    model.train()
    count_loss = float('inf')
    out = model(data)
    # centers = model.centers
    # k_input = out.detach().cpu().numpy()

    loss = infoNCE_loss(out,topk,bottomk)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss,out

def train_no_info_model(data,centers):
    model.train()
    count_loss = float('inf')
    out = model(data)
    # centers = model.centers
    # k_input = out.detach().cpu().numpy()
    out = out[learing_data_indices]

    loss = KL_loss(graph_emb=out,centers=centers)[0]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss,out

def pre_trian_model(data):
    model.train()
    out = model(data)

    edge_index = data.edge_index
    num_nodes = data.num_nodes

    adj_matrix_sparse = torch.sparse_coo_tensor(
        indices=edge_index,
        values=torch.ones(edge_index.shape[1], device=device),
        size=(num_nodes, num_nodes)
    )

    adj_matrix_dense = adj_matrix_sparse.to_dense()
    pre_loss = reconstruction_loss(adj_matrix_dense,out)
    optimizer.zero_grad()
    pre_loss.backward()
    optimizer.step()

    return pre_loss


@torch.no_grad()
def test_model(data):
    model.eval()
    # model.load_state_dict(torch.load('clustering_model_state_dict.pth'))
    embeddings = model(data)
    # loss = infoNCE_loss(embeddings, topk, bottomk,centers)
    return embeddings[learing_data_indices],embeddings

def min_max_normalize(tensor, min_value=0.0, max_value=1.0):
    """
    对输入张量进行Max-Min归一化，将其缩放到指定范围内。

    参数:
    tensor (torch.Tensor): 需要归一化的张量
    min_value (float): 归一化后的最小值
    max_value (float): 归一化后的最大值

    返回:
    torch.Tensor: 归一化后的张量
    """
    tensor_min = tensor.min()
    tensor_max = tensor.max()

    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    normalized_tensor = normalized_tensor * (max_value - min_value) + min_value

    return normalized_tensor

def plot_clusters(X, labels, true_labels, centroids,test_mask):
    colors = ['blue', 'green']
    markers = ['o', '^']

    plt.figure(figsize=(8, 6))

    for label in np.unique(labels):
        for true_label in np.unique(true_labels):
            mask = (labels == label) & (true_labels == true_label)
            cluster_points = X[test_mask][mask]
            plt.scatter(
                cluster_points[:, 0], cluster_points[:, 1],
                c=colors[label], marker=markers[true_label],
                label=f'Cluster {label}, True {true_label}'
            )

    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='x', label='Centroids')

    plt.title(f'{args.dataset} K-means Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def plot_clusters_4(X, labels, true_labels, centroids, test_mask, dataset_name):
    # 增加更多的颜色和标记以支持4个类别
    colors = ['blue', 'green', 'orange', 'purple']
    markers = ['o', '^', 's', 'D']

    plt.figure(figsize=(8, 6))

    # 绘制数据点，根据标签和真实标签组合进行绘制
    for label in np.unique(labels):
        for true_label in np.unique(true_labels):
            mask = (labels == label) & (true_labels == true_label)
            cluster_points = X[test_mask][mask]
            plt.scatter(
                cluster_points[:, -1], cluster_points[:, 0],
                c=colors[label], marker=markers[true_label],
                label=f'Cluster {label}, True {true_label}'
            )

    # 绘制中心点
    plt.scatter(centroids[:, -1], centroids[:, 0], s=300, c='red', marker='x', label='Centroids')

    # 设置标题和标签
    plt.title(f'{dataset_name} K-means Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def visualize_clustering(true_labels, cluster_results):
    """
    Visualize the clustering results against the true labels.

    Parameters:
        true_labels (list or np.array): The ground truth labels (0 or 1).
        cluster_results (list or np.array): The clustering results (0 or 1).
    """
    if len(true_labels) != len(cluster_results):
        raise ValueError("The lengths of true_labels and cluster_results must be the same.")

    # Convert to numpy arrays for easier processing
    true_labels = np.array(true_labels)
    cluster_results = np.array(cluster_results)

    # Create a figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    # Plot the true labels
    axes[0].scatter(range(len(true_labels)), [0] * len(true_labels), c=true_labels, cmap='coolwarm', s=50, label='真实情况')
    axes[0].set_title("真实结果")
    axes[0].get_yaxis().set_visible(False)
    axes[0].legend()

    # Plot the clustering results
    axes[1].scatter(range(len(cluster_results)), [0] * len(cluster_results), c=cluster_results, cmap='coolwarm', s=50, label='聚类情况')
    axes[1].set_title("聚类结果")
    axes[1].get_yaxis().set_visible(False)
    axes[1].legend()

    # Set the x-axis label
    plt.xlabel("数据点")

    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    """#########  Cluster control  ############"""
    cluster_number = 2
    centriods = torch.rand((cluster_number,128),dtype=torch.float).to(device)

    """#########  training control ###########"""
    # pre_train_flag = int(sys.argv[4])
    pre_train_flag = 1
    train_flag = 1

    '''################  Feature reconstruction #############'''
    data.x = torch.cat([data.x, data.added_fea.view(-1, 1)], dim=-1)
    # data.x = data.added_fea
    """###############   Initial model ##########"""
    if args.type == "Binary":
        # model = GATemb(in_channels=1,out_channels=2,hidden_channels=128).to(device)
        model = GCNemb(num_node_features=data.x.shape[1],output_dim=2,hidden_dim=128,cluster_num=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        # optimizer = optim.SGD(model.parameters(),lr=args.learning_rate, weight_decay=args.weight_decay)
    print(model)

    """##############  Sample construction  ###############"""
    # topk, bottomk = supervised_select_samples_based_on_edges()
    topk, bottomk = select_samples_based_on_Reg()
    max_node_id = max(topk.keys())  # 确保包含所有节点
    topk_list = [topk[i] for i in range(max_node_id + 1)]
    bottomk_list = [bottomk[i] for i in range(max_node_id + 1)]
    topk = torch.tensor(topk_list, dtype=torch.long,device=device)
    bottomk = torch.tensor(bottomk_list, dtype=torch.long,device=device)

    count_loss = float('inf')

    mode = 'kmeans'


    if pre_train_flag == 1:
            for i in range(100):
                pre_loss = pre_trian_model(data)
                print(f'pretrain loss: {pre_loss}')
                if count_loss > pre_loss:
                    torch.save(model.state_dict(),'pre_train_model.pth')
                    count_loss = pre_loss
    model.load_state_dict(torch.load('pre_train_model.pth'))
    if train_flag == 1:
        best_nmi = float('-inf')
        for i in range(100):
            if mode == 'kmeans':
                loss,out = train_clustering_model(data,topk,bottomk,centers=centriods)
                k_input = out[learing_data_indices].detach().cpu().numpy()
                kmeans = KMeans(n_clusters=cluster_number)
                kmeans.fit(k_input)
                centriods = torch.tensor(kmeans.cluster_centers_).to(device)
                cluster_results = kmeans.fit_predict(k_input)
            elif mode  == 'no_kl':
                loss, out = train_no_kl_model(data, topk, bottomk)
                k_input = out[learing_data_indices].detach().cpu().numpy()
                kmeans = KMeans(n_clusters=cluster_number)
                kmeans.fit(k_input)
                cluster_results = kmeans.fit_predict(k_input)
            elif mode  == 'no_info':
                loss,out = train_no_info_model(data,centers=centriods)
                k_input = out.detach().cpu().numpy()
                kmeans = KMeans(n_clusters=cluster_number)
                kmeans.fit(k_input)
                centriods = torch.tensor(kmeans.cluster_centers_).to(device)
                cluster_results = kmeans.fit_predict(k_input)
            else:
                loss,out = train_no_kl_model(data, topk, bottomk)
                s_input = out[learing_data_indices].detach().cpu().numpy()
                spec_cluster = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans')
                cluster_results = spec_cluster.fit_predict(s_input)

            train_nmi = normalized_mutual_info_score(data.label.cpu().numpy()[learing_data_indices],cluster_results)
            train_acc = clustering_accuracy(data.label.cpu().numpy()[learing_data_indices],cluster_results)
            print(f"epoch: {color_green}{i}{color_reset}, total_loss: {color_red}{loss}{color_reset}"
                  f", ACC: {color_red}{train_acc}{color_reset}, NMI: {color_red}{train_nmi}{color_reset}"
                  )
            if best_nmi < train_nmi:
                torch.save(model.state_dict(), 'best_clustering_model.pth')
                best_nmi = train_nmi
        model.load_state_dict(torch.load('best_clustering_model.pth'))
        emd,to_plot = test_model(data)
        emd = emd.cpu().numpy()


        """############# Downstream task ##############"""
        if mode == 'spec':
            clustering_method = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans')
            predict_labels = clustering_method.fit_predict(emd)
            print(args.contrastive_mode)
        else:
            kmeans = KMeans(n_clusters=cluster_number)
            predict_labels = kmeans.fit_predict(emd)
            cen = kmeans.cluster_centers_
            print(args.contrastive_mode)

            # print(sys.argv[1])
        nmi = normalized_mutual_info_score(data.label.cpu().numpy()[learing_data_indices],predict_labels)
        acc = clustering_accuracy(data.label.cpu().numpy()[learing_data_indices],predict_labels)
        ari = adjusted_rand_score(data.label.cpu().numpy()[learing_data_indices],predict_labels)
        f1 = f1_score(data.label.cpu().numpy()[learing_data_indices],predict_labels)
        print(f"ACC: {acc}")
        print(f"NMI: {nmi}")
        print(f'ARI: {ari}')
        print(f'F1: {f1}')




