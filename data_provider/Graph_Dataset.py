# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import gc

def buildGraph(x:torch.tensor, sigma:float):
    """
    X是输入样本矩阵，N*d，N为样本个数，d为样本特征维度；
    sigma是高斯核函数参数
    返回一个图的邻接矩阵，矩阵第i行第j列元素为相似权重w_ij，矩阵形状为N*N
    """
    # 计算邻接图的权重
    x_tile = x.unsqueeze(1).expand(x.shape[0],x.shape[0],x.shape[1]).detach()
    x_tile = x_tile - x.unsqueeze(1).transpose(0,1)
    x_distance = torch.mul(x_tile, x_tile).sum(dim=-1)
    x_distance = torch.mul(x_distance, -1/(2*sigma**2))
    laplacian_weight = torch.exp(x_distance)
    weight_diag = torch.diag(laplacian_weight)
    laplacian_diag = torch.diag_embed(weight_diag)
    laplacian_weight = laplacian_weight - laplacian_diag
    return laplacian_weight
    


def buildGraph_lessmem(x:torch.tensor, sigma:float):
    x_distance = []
    for x_sample in x:
        x_diff = x_sample.unsqueeze(0) - x.unsqueeze(0)
        x_dist = torch.mul(x_diff, x_diff).sum(dim=-1)
        x_dist = torch.mul(x_dist, -1/(2*sigma**2))
        x_distance.append(torch.exp(x_dist))
    laplacian_weight = torch.cat(x_distance, dim=0)
    weight_diag = torch.diag(laplacian_weight)
    laplacian_diag = torch.diag_embed(weight_diag)
    laplacian_weight = laplacian_weight - laplacian_diag
    return laplacian_weight



class GraphSampling_Dataset(Dataset):
    def __init__(self, dataset_cfg: dict):
        self.label_rate = dataset_cfg['label_rate']
        label_s = dataset_cfg['labeled_data_step']
        self.weight_graph = dataset_cfg['weight_graph']
        self.sample_num = dataset_cfg['sample_num']
        input_data = np.genfromtxt(fname = dataset_cfg['root'] + dataset_cfg['name'], delimiter = ',')
        train_data = input_data[dataset_cfg['train_start']:dataset_cfg['train_end'], dataset_cfg['start_feature']:dataset_cfg['end_feature']]
        gdb_data = input_data[dataset_cfg['start_seg']:dataset_cfg['end_seg'], dataset_cfg['start_feature']:dataset_cfg['end_feature']]
        gdb_label = input_data[dataset_cfg['start_seg']:dataset_cfg['end_seg'], dataset_cfg['target']]
        # 针对脱丁烷塔进行特征工程
        if dataset_cfg['debutanizer'] == True:
            input_data_temp = input_data[4:2394,:5]
            x_5 = np.expand_dims(input_data[:,4],1)
            x_9 = np.expand_dims(((input_data[4:2394, 5]+input_data[4:2394, 6])/2),1)
            y = np.expand_dims(input_data[:,7],1)
            input_data = np.concatenate((input_data_temp, x_5[3:2393], x_5[2:2392], x_5[1:2391], x_9,
                                   y[3:2393],y[2:2392],y[1:2391],y[:2390]),1)
            input_label = y[4:2394]
            train_data = input_data[dataset_cfg['train_start']:dataset_cfg['train_end']]
            gdb_data = input_data[dataset_cfg['start_seg']:dataset_cfg['end_seg']]
            gdb_label = input_label[dataset_cfg['start_seg']:dataset_cfg['end_seg']]        
        # 变量标准化
        self.x_scaler = dataset_cfg['x_scaler']
        self.y_scaler = dataset_cfg['y_scaler']
        if dataset_cfg['scale']==True:
            self.x_scaler.fit(train_data)
            if dataset_cfg['y_scale']==True:
                self.y_scaler.fit(input_data[dataset_cfg['train_start']:dataset_cfg['train_end'], dataset_cfg['target']].reshape(-1,1))
        if dataset_cfg['debutanizer'] == False and dataset_cfg['scale'] == True:
            gdb_data = self.x_scaler.transform(gdb_data)
            if dataset_cfg['y_scale']==True:
                gdb_label = self.y_scaler.transform(gdb_label.reshape(-1,1))
        # 选取数据样本
        # 有标签样本直接采样，无标签数据样本采样的原则是尽量取采样时刻上接近有标签样本的无标签样本
        num_unlabel_per_interval = int(1/self.label_rate)
        if num_unlabel_per_interval%2==0:
            left_idx = -int(num_unlabel_per_interval/2)+1
            right_idx = int(num_unlabel_per_interval/2)
        else:
            left_idx = -int((num_unlabel_per_interval-1)/2)
            right_idx = int((num_unlabel_per_interval-1)/2)
        label_idx = int((num_unlabel_per_interval-1)/2)
        unlabeled_data = []
        label = []
        label_ind_list = [] # 使用一个表来标记有标签数据样本
        for i in np.arange(gdb_data.shape[0]):
            if i%label_s == 0:
                if num_unlabel_per_interval==1:
                    unlabeled_data.append(gdb_data[i])
                    label.append(np.array(gdb_label[i], dtype=np.float))
                    label_ind_list.append(1)
                else:
                    unlabeled_data.append(gdb_data[i+left_idx:i+right_idx+1])
                    label.append(gdb_label[i+left_idx:i+right_idx+1])
                    temp_ind=[0]*(right_idx-left_idx+1)
                    temp_ind[label_idx] = 1
                    label_ind_list.append(temp_ind)
        self.data = torch.tensor(np.concatenate(unlabeled_data,
                                          axis=0).reshape(-1,gdb_data.shape[1]), dtype=torch.float32)
        self.label = torch.tensor(np.concatenate(label, axis=0).reshape(-1,1), dtype=torch.float32)
        self.label_ind = np.concatenate(label_ind_list, axis=0).reshape(-1,1)
        # 准备样本的权重
        self.sorted_weight, self.weight_index = torch.sort(self.weight_graph, dim=-1, descending=True)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        # 如果有标签样本邻近点也是有标签样本，则不计权重；其余情况计算权重
        if self.label_ind[idx]==1:
            neighbor_idx = self.weight_index[idx, 0:self.sample_num]
            neighbor = self.data[neighbor_idx]
            neighbor_label = torch.tensor([self.label[neighbor_idx[i]] if self.label_ind[neighbor_idx[i]]==1 else -1 
                                           for i in range(neighbor.shape[0])], dtype=torch.float32)
            neighbor_weight = torch.tensor([self.sorted_weight[idx,i] if self.label_ind[neighbor_idx[i]]==0 else 0 
                                           for i in range(neighbor.shape[0])], dtype=torch.float32)
            return (self.data[idx].float(), self.label[idx], neighbor.float(), neighbor_label, neighbor_weight.float())
        else:
            neighbor_idx = self.weight_index[idx, 0:self.sample_num]
            neighbor = self.data[neighbor_idx]
            neighbor_label = torch.tensor([self.label[neighbor_idx[i]] if self.label_ind[neighbor_idx[i]]==1 else -1 
                                           for i in range(neighbor.shape[0])], dtype=torch.float32)
            neighbor_weight = self.sorted_weight[idx, 0:self.sample_num]
            return (self.data[idx].float(), torch.tensor([-1], dtype=torch.float32), neighbor.float(), neighbor_label, neighbor_weight.float())