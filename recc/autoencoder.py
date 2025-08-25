import torch.nn
import random
import pandas as pd
import sys,os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from utils import *
import itertools
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics import normalized_mutual_info_score ,adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)  # 设置CPU生成随机数的种子，方便下次复现实验结果。
torch.cuda.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("cuda ", torch.cuda.is_available())

numfea = 3
data = load_data(args.dataset,numfea)
data.to(device)

total_data_mask = data.train_mask + data.test_mask

def clustering_accuracy(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    accuracy = sum(w[row_ind[i], col_ind[i]] for i in range(len(row_ind))) / y_pred.size
    return accuracy

def KL_loss(graph_emb,centers):
    # q_ij_matrix = torch.zeros((graph_emb.shape[0],centers.shape[0]),dtype=torch.float)
    # for i in range(graph_emb.shape[0]):
    #     for j in range(centers.shape[0]):
    #         q_down = 0
    #         for j_ in range(centers.shape[0]):
    #             q_down +=  torch.norm(graph_emb[i] - centers[j_],p=2)
    #         q_up = 1 / (torch.norm(graph_emb[i] - centers[j],p=2) + 1)
    #         q_ij =  q_up /  (1 /q_down)
    #         q_ij_matrix[i,j]=q_ij
    # f_j = torch.sum(q_ij_matrix,dim=0,keepdim=True)
    # p_down= torch.sum((q_ij_matrix ** 2 / f_j), dim=1,keepdim=True)
    # p_up = q_ij_matrix ** 2 / f_j
    # p_ij_matrix = p_up / p_down.expand_as(p_up)
    # kl_loss =torch.sum( p_ij_matrix * torch.log(p_ij_matrix / q_ij_matrix) )
    # return kl_loss
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
class encoder(torch.nn.Module):
    def __init__(self,input_dim,output_dim):
        super(encoder, self).__init__()
        self.mlp1 = torch.nn.Linear(input_dim,64)
        self.mlp2 = torch.nn.Linear(64,8)
        self.mlp3 = torch.nn.Linear(8,8)
        self.mlp4 = torch.nn.Linear(8,128)
        self.mlp5 = torch.nn.Linear(128,output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self,data):
        x = data
        x = self.relu(self.mlp1(x))
        x = self.relu(self.mlp2(x))
        x = self.relu(self.mlp3(x))
        x = self.relu(self.mlp4(x))
        x = self.mlp5(x)
        return x

class decoder(torch.nn.Module):
    def __init__(self,input_dim,output_dim):
        super(decoder, self).__init__()
        self.mlp1 = torch.nn.Linear(input_dim,128)
        self.mlp2 = torch.nn.Linear(128,8)
        self.mlp3 = torch.nn.Linear(8,8)
        self.mlp4 = torch.nn.Linear(8,64)
        self.mlp5 = torch.nn.Linear(64,output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self,data):
        x = data
        x = self.relu(self.mlp1(x))
        x = self.relu(self.mlp2(x))
        x = self.relu(self.mlp3(x))
        x = self.relu(self.mlp4(x))
        x = self.mlp5(x)
        return x


def compute_directional_centrality(data, k=10):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data)
    _, indices = nbrs.kneighbors(data)

    directional_centrality = []
    for i in range(len(data)):
        neighbors = data[indices[i][1:]]
        vectors = neighbors - data[i]

        mean_direction = np.mean(vectors, axis=0)
        norm = np.linalg.norm(mean_direction)

        if norm > 0:
            mean_direction /= norm

        centrality = np.sum(mean_direction)
        directional_centrality.append(centrality)

    return np.array(directional_centrality)

def ldcca_clustering(H,n_clusters):

    kmeans = KMeans(n_clusters=H.shape[1], random_state=42)
    labels = kmeans.fit_predict(H)
    return labels

    return labels


def LCCF(X, n_clusters, alpha, max_iter=100, tol=1e-4):

    m, n = X.shape
    H = np.random.rand(m, n_clusters)
    W = np.random.rand(n_clusters, n)
    S = np.exp(-cdist(X, X, 'euclidean') ** 2)
    np.fill_diagonal(S, 0)

    for iteration in range(max_iter):

        numerator = X @ W.T + alpha * S @ H
        denominator = H @ (W @ W.T) + alpha * (np.sum(S, axis=1, keepdims=True) * H) + 1e-9
        H *= numerator / denominator

        numerator = H.T @ X
        denominator = H.T @ H @ W + 1e-9
        W *= numerator / denominator

        reconstruction_error = np.linalg.norm(X - H @ W, 'fro')
        if reconstruction_error < tol:
            break

    return H, W

def l2_loss(x,y):
    return torch.norm((x-y),p=2)
def pre_train():
    encoder_model.train()
    decoder_model.train()
    embeddings = encoder_model(data.x)
    reconstrution = decoder_model(embeddings)
    loss = l2_loss(data.x,reconstrution)

    pre_optim.zero_grad()
    loss.backward()
    pre_optim.step()

    return embeddings,loss

def train():
    encoder_model.load_state_dict(torch.load('E:\Documents\GNNS\GNNS\CL\saved_model\\auto_encoder.pth'))
    encoder_model.train()
    embeddings = encoder_model(data.x)
    loss = KL_loss(graph_emb=embeddings[total_data_mask],centers=centers)[0]
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss

def test():
    encoder_model.eval()
    embeddings = encoder_model(data.x)
    p_ij_matrix = KL_loss(graph_emb=embeddings[total_data_mask],centers=centers)[1]
    return embeddings,p_ij_matrix


if __name__ == '__main__':

    """
        Initial settings 
    """
    encoder_model = encoder(data.x.shape[1],2).to(device)
    decoder_model = decoder(2,data.x.shape[1]).to(device)
    cluster_number = 2
    kmeans = KMeans(n_clusters=2)
    init_cluster = torch.tensor(kmeans.fit_predict(data.x[:, :cluster_number].cpu().numpy())).to(device)
    centers = torch.tensor(kmeans.cluster_centers_).to(device)
    pre_optim = torch.optim.SGD(itertools.chain(encoder_model.parameters(), decoder_model.parameters()),
                                lr=args.learning_rate, weight_decay=args.weight_decay)
    optim = torch.optim.SGD(itertools.chain(encoder_model.parameters(), centers), lr=args.learning_rate,
                            weight_decay=args.weight_decay)

    """  
        Training control   
    """
    pre_train_flag = 1
    train_flag = 1

    # for method in ['DEC','DCN','NMF+KM','LCCF']:
    for method in ['LCCF']:
        ave_acc,ave_nmi,ave_ari = 0,0,0
        if method == 'DEC':
            if pre_train_flag == 1:
                for i in range(500):
                    _,loss = pre_train()
                    print(f'pretraining loss: {loss}')
                torch.save(encoder_model.state_dict(),'E:\Documents\GNNS\GNNS\CL\saved_model\\auto_encoder.pth')

            if train_flag == 1:
                for i in range(100):
                    kl_loss = train()
                    print(f'kl loss: {kl_loss}')

            embeddings,p_ij_matrix = test()
            predicts = torch.argmax(p_ij_matrix,dim=1).cpu().numpy()
            acc = clustering_accuracy(data.label[total_data_mask],predicts)
            NMI = normalized_mutual_info_score(data.label[total_data_mask].cpu().numpy(),predicts)
            ARI = adjusted_rand_score(data.label[total_data_mask].cpu().numpy(),predicts)
            print(args.dataset)
            print(f'cluster ACC: {acc}, NMI: {NMI}, ARI: {ARI}')
        if method == 'DCN':
            optim = torch.optim.SGD(itertools.chain(encoder_model.parameters(),decoder_model.parameters(),centers), lr=args.learning_rate,
                                weight_decay=args.weight_decay)
            for i in range(1000):
                encoder_model.train()
                decoder_model.train()
                embeddings = encoder_model(data.x)
                reconstrution = decoder_model(embeddings)
                loss = l2_loss(data.x, reconstrution)
                embeddings = embeddings[total_data_mask]

                distances = torch.sqrt(((embeddings[:, None, :] - centers[None, :, :]) ** 2).sum(dim=2))
                cluster_indices = torch.argmin(distances,dim=1)
                selected_centers = centers[cluster_indices]
                c_loss = torch.norm(embeddings-selected_centers)

                loss = loss + c_loss
                optim.zero_grad()
                loss.backward()
                optim.step()
                print(f'training loss: {loss}')
            encoder_model.eval()
            embeddings = encoder_model(data.x)
            embeddings = embeddings[total_data_mask]
            distances = torch.sqrt(((embeddings[:, None, :] - centers[None, :, :]) ** 2).sum(dim=2))
            cluster_indices = torch.argmin(distances, dim=1)
            predicts = cluster_indices.cpu().numpy()
            acc = clustering_accuracy(data.label[total_data_mask],predicts)
            NMI = normalized_mutual_info_score(data.label[total_data_mask].cpu().numpy(),predicts)
            ARI = adjusted_rand_score(data.label[total_data_mask].cpu().numpy(),predicts)
            print(args.dataset)
            print(f'cluster ACC: {acc}, NMI: {NMI}, ARI: {ARI}')
        if method == 'NMF+KM':
            edge_index = data.edge_index
            num_nodes = data.num_nodes

            adj_matrix_sparse = torch.sparse_coo_tensor(
                indices=edge_index,
                values=torch.ones(edge_index.shape[1], device=device),
                size=(num_nodes, num_nodes)
            )
            adj_matrix_dense = adj_matrix_sparse.to_dense()

            X = adj_matrix_dense.cpu().numpy()
            nmf = NMF(n_components=10, init='random', random_state=args.seed)
            W = nmf.fit_transform(X)
            H = nmf.components_
            kmeans = KMeans(n_clusters=5, random_state=0).fit(W)
            predicts = kmeans.labels_
            acc = clustering_accuracy(data.label[total_data_mask],predicts[total_data_mask])
            NMI = normalized_mutual_info_score(data.label[total_data_mask].cpu().numpy(),predicts[total_data_mask])
            ARI = adjusted_rand_score(data.label[total_data_mask].cpu().numpy(),predicts[total_data_mask])
            print(args.dataset)
            print(f'cluster ACC: {acc}, NMI: {NMI}, ARI: {ARI}')
        if method == 'LCCF':
            H, W = LCCF(data.x[total_data_mask].cpu().numpy(), n_clusters=2, alpha=0.1)
            predicts = ldcca_clustering(H,n_clusters=2)
            acc = clustering_accuracy(data.label[total_data_mask].cpu().numpy(),predicts)
            NMI = normalized_mutual_info_score(data.label[total_data_mask].cpu().numpy(),predicts)
            ARI = adjusted_rand_score(data.label[total_data_mask].cpu().numpy(),predicts)
            print(args.dataset)
            print(f'cluster ACC: {acc}, NMI: {NMI}, ARI: {ARI}')
        ave_acc = ave_acc + acc
        ave_nmi = ave_nmi + NMI
        ave_ari = ave_ari + ARI
        columns = ['method', 'ACC', 'NMI', "ARI"]
        records = pd.DataFrame(columns=columns)
        records = records.append({'method': method, 'ACC': ave_acc/5, 'NMI': ave_nmi/5, 'ARI': ave_ari/5}, ignore_index=True)
        # records.to_csv(f'D:\\final_res\\compared\\{sys.argv[1]}.csv', mode='a', index=False)