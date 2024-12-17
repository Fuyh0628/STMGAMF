from __future__ import division
from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')
import torch.optim as optim
from utils import *
from models import STMGAMF
import os
import argparse
from config import Config
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, homogeneity_completeness_v_measure


def load_data(dataset):
    print("load data:")
    path = "../generate_data/DLPFC/" + dataset + "/STMGAMF.h5ad"
    adata = sc.read_h5ad(path)
    features = torch.FloatTensor(adata.X)
    labels = adata.obs['ground']
    fadj = adata.obsm['fadj']
    sadj = adata.obsm['sadj']
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    graph_nei = torch.LongTensor(adata.obsm['graph_nei'])
    graph_neg = torch.LongTensor(adata.obsm['graph_neg'])
    print("done")
    return adata, features, labels, nfadj, nsadj, graph_nei, graph_neg

def train():
    model.train()
    optimizer.zero_grad()
    com1, com2, emb, pi, disp, mean = model(features, sadj, fadj)
    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
    reg_loss = regularization_loss(emb, graph_nei, graph_neg)
    con_loss = consistency_loss(com1, com2)
    total_loss = config.alpha * zinb_loss + config.beta * con_loss + config.gamma * reg_loss
    emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
    mean = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values
    total_loss.backward()
    optimizer.step()
    return emb, mean, zinb_loss, reg_loss, con_loss, total_loss

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    datasets = ['151676']
    for i in range(len(datasets)):
        dataset = datasets[i]
        config_file = f'./config/DLPFC_{dataset}_best.ini'
        print(dataset)
        adata, features, labels, fadj, sadj, graph_nei, graph_neg = load_data(dataset)
        print(adata)

        plt.rcParams["figure.figsize"] = (3, 3)
        savepath = './result/DLPFC/' + dataset + '/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        title = "Manual annotation (slice #" + dataset + ")"
        sc.pl.spatial(adata, img_key="hires", color=['ground_truth'], title=title, show=False)
        plt.savefig(savepath + 'Manual Annotation.jpg', bbox_inches='tight', dpi=600)
        plt.draw()
        plt.pause(1)
        plt.close()

        config = Config(config_file)
        cuda = not config.no_cuda and torch.cuda.is_available()
        use_seed = not config.no_seed

        _, ground = np.unique(np.array(labels, dtype=str), return_inverse=True)
        ground = torch.LongTensor(ground)
        config.n = len(ground)
        config.class_num = len(ground.unique())

        config.epochs = config.epochs + 1

        if cuda:
            features = features.cuda()
            sadj = sadj.cuda()
            fadj = fadj.cuda()
            graph_nei = graph_nei.cuda()
            graph_neg = graph_neg.cuda()

        import random
        config.sadj = sadj.coalesce().indices().cpu().numpy()
        config.fadj = fadj.coalesce().indices().cpu().numpy()
        config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        np.random.seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        os.environ['PYTHONHASHSEED'] = str(config.seed)
        if not config.no_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

        model = STMGAMF(nfeat=config.fdim, nhid1=config.nhid1, nhid2=config.nhid2, dropout=config.dropout, config=config)
        print(model)
        if cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        epoch_max = 0
        ari_max = 0
        nmi_max = 0
        homogeneity_max = 0
        completeness_max = 0
        v_measure_max = 0
        idx_max = []
        mean_max = []
        emb_max = []
        aris_list = []
        nmis_list = []

        for epoch in range(config.epochs):
            emb, mean, zinb_loss, reg_loss, con_loss, total_loss = train()
            print(dataset, ' epoch: ', epoch, ' zinb_loss = {:.2f}'.format(zinb_loss),
                  ' reg_loss = {:.2f}'.format(reg_loss), ' con_loss = {:.2f}'.format(con_loss),
                  ' total_loss = {:.2f}'.format(total_loss))
            kmeans = KMeans(n_clusters=config.class_num).fit(emb)
            idx = kmeans.labels_

            ari_res = metrics.adjusted_rand_score(labels, idx)
            nmi_res = normalized_mutual_info_score(labels, idx)
            Homogeneity, Completeness, V_measure = homogeneity_completeness_v_measure(labels, idx)

            if ari_res > ari_max:
                ari_max = ari_res
                nmi_max = nmi_res
                Homogeneity_max = Homogeneity
                Completeness_max = Completeness
                V_measure_max = V_measure
                epoch_max = epoch
                idx_max = idx
                mean_max = mean
                emb_max = emb


            print(f"epoch: {epoch}, ARI: {ari_res}, NMI: {nmi_res}, Best ARI: {ari_max}")

        title1 = f'{dataset}: ARI={ari_max:.2f}, NMI={nmi_max:.2f}'
        title2 = f'{dataset}: Homogeneity={Homogeneity:.2f}, Completeness={Completeness:.2f}, V-measure={V_measure:.2f}'

        adata.obs['idx'] = idx_max.astype(str)
        adata.obsm['emb'] = emb_max
        adata.obsm['mean'] = mean_max

        print(f"Best ARI: {ari_max}, Corresponding NMI: {nmi_max}, Homogeneity: {Homogeneity}, Completeness: {Completeness}, V-measure: {V_measure}")

        sc.pl.spatial(adata, img_key="hires", color=['idx'], title=title1, show=False)
        plt.savefig(savepath + 'STMGAMF.jpg', bbox_inches='tight', dpi=600)
        plt.show()

        sc.pp.neighbors(adata, use_rep='mean')
        sc.tl.umap(adata)
        plt.rcParams["figure.figsize"] = (3, 3)
        sc.tl.paga(adata, groups='idx')
        sc.pl.paga_compare(adata, legend_fontsize=8, frameon=False, size=20, title=title2, legend_fontoutline=2,
                           show=False)
        plt.savefig(savepath + 'STMGAMF_umap_mean.jpg', bbox_inches='tight', dpi=600)
        plt.show()

        pd.DataFrame(emb_max).to_csv(savepath + 'STMGAMF_emb.csv')
        pd.DataFrame(idx_max).to_csv(savepath + 'STMGAMF_idx.csv')
        adata.layers['X'] = adata.X
        adata.layers['mean'] = mean_max
        adata.write(savepath + 'STMGAMF.h5ad')

