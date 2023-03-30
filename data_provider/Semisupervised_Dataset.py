# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset
import gc

class TrainUnlabelData(Dataset):
    def __init__(self, dataset_cfg : dict):
        self.label_rate = dataset_cfg['label_rate']
        label_s = dataset_cfg['labeled_data_step']
        input_data = np.genfromtxt(fname = dataset_cfg['root'] + dataset_cfg['name'], delimiter = ',')
        train_data = input_data[dataset_cfg['train_start']:dataset_cfg['train_end'], dataset_cfg['start_feature']:dataset_cfg['end_feature']]
        gdb_data = input_data[dataset_cfg['start_seg']:dataset_cfg['end_seg'], dataset_cfg['start_feature']:dataset_cfg['end_feature']]
        gdb_label = input_data[dataset_cfg['start_seg']:dataset_cfg['end_seg'], dataset_cfg['target']]
        # 变量标准化函数初始化
        self.x_scaler = dataset_cfg['x_scaler']
        self.y_scaler = dataset_cfg['y_scaler']
        if dataset_cfg['scale']==True:
            self.x_scaler.fit(train_data)
            if dataset_cfg['y_scale']==True:
                self.y_scaler.fit(input_data[dataset_cfg['train_start']:dataset_cfg['train_end'], dataset_cfg['target']].reshape(-1,1))
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
            
        # 选取数据样本
        # 有标签样本直接采样，无标签数据样本采样的原则是尽量取采样时刻上接近有标签样本的无标签样本
        num_unlabel_per_interval = int(1/self.label_rate)
        if num_unlabel_per_interval%2==0:
            left_idx = -int(num_unlabel_per_interval/2)+1
            right_idx = int(num_unlabel_per_interval/2)
        else:
            left_idx = -int((num_unlabel_per_interval-1)/2)
            right_idx = int((num_unlabel_per_interval-1)/2)
        unlabeled_data = []
        label = []
        for i in np.arange(gdb_data.shape[0]):
            if i%label_s == 0:
                if num_unlabel_per_interval==1:
                    unlabeled_data.append(gdb_data[i])
                    label.append(np.array(gdb_label[i], dtype=np.float))
                else:
                    unlabeled_data.append(gdb_data[i+left_idx:i+right_idx+1])
                    label.append(gdb_label[i+left_idx:i+right_idx+1])
        self.data = torch.tensor(np.concatenate(unlabeled_data,
                                          axis=0).reshape(-1,gdb_data.shape[1]), dtype=torch.float32)
        self.label = torch.tensor(np.concatenate(label, axis=0).reshape(-1,1), dtype=torch.float32)
        # 变量标准化
        if dataset_cfg['debutanizer']==False and dataset_cfg['scale']==True:
            self.data = self.x_scaler.transform(self.data)
            if dataset_cfg['y_scale']==True:
                self.label = self.y_scaler.transform(self.label.reshape(-1,1))     
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return (self.data[idx],self.label[idx])



class TrainUnlabelDataOnly(Dataset):
    def __init__(self, dataset_cfg : dict):
        self.label_rate = dataset_cfg['label_rate']
        label_s = dataset_cfg['labeled_data_step']
        input_data = np.genfromtxt(fname = dataset_cfg['root'] + dataset_cfg['name'], delimiter = ',')
        train_data = input_data[dataset_cfg['train_start']:dataset_cfg['train_end'], 
                                dataset_cfg['start_feature']:dataset_cfg['end_feature']]
        gdb_data = input_data[dataset_cfg['start_seg']:dataset_cfg['end_seg'], 
                             dataset_cfg['start_feature']:dataset_cfg['end_feature']]
        gdb_label = input_data[dataset_cfg['start_seg']:dataset_cfg['end_seg'], dataset_cfg['target']]
        # 变量标准化函数初始化
        self.x_scaler = dataset_cfg['x_scaler']
        if dataset_cfg['scale']==True:
            self.x_scaler.fit(train_data)
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
        # 选取无标签数据样本
        # 无标签数据样本采样的原则是尽量取采样时刻上接近有标签样本的无标签样本
        num_unlabel_per_interval = int(1/self.label_rate)
        if num_unlabel_per_interval%2==0:
            left_idx = -int(num_unlabel_per_interval/2)+1
            right_idx = int(num_unlabel_per_interval/2)
        else:
            left_idx = -int((num_unlabel_per_interval-1)/2)
            right_idx = int((num_unlabel_per_interval-1)/2)
        del_idx = int((num_unlabel_per_interval-1)/2)
        unlabeled_data = []
        label = []
        for i in np.arange(gdb_data.shape[0]):
            if i%label_s == 0 and i>0 and i+right_idx+1<=gdb_data.shape[0]:
                if num_unlabel_per_interval==1:
                    unlabeled_data.append(gdb_data[i])
                    label.append(np.array(gdb_label[i], dtype=np.float))
                else:
                    unlabeled_data.append(np.delete(gdb_data[i+left_idx:i+right_idx+1], del_idx, 0))
                    label.append(np.delete(gdb_label[i+left_idx:i+right_idx+1],del_idx,0))
        self.data = torch.tensor(np.concatenate(unlabeled_data, 
                                          axis=0).reshape(-1,gdb_data.shape[1]), dtype=torch.float32)
        self.label = torch.tensor(np.concatenate(label, axis=0).reshape(-1,1), dtype=torch.float32)
        # 变量标准化
        if dataset_cfg['debutanizer']==False and dataset_cfg['scale']==True:
            self.data = self.x_scaler.transform(self.data)
        
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return (self.data[idx],self.label[idx])



class TrVldLabelData(Dataset):
    def __init__(self,dataset_cfg:dict):
        label_s = dataset_cfg['labeled_data_step']
        input_data = np.genfromtxt(fname = dataset_cfg['root'] + dataset_cfg['name'], delimiter = ',')
        train_data = input_data[dataset_cfg['train_start']:dataset_cfg['train_end'], 
                               dataset_cfg['start_feature']:dataset_cfg['end_feature']]
        data_mean = train_data.mean(axis = 0).reshape(1, -1)
        data_var = train_data.var(axis = 0).reshape(1, -1)
        gdb_data = input_data[dataset_cfg['start_seg']:dataset_cfg['end_seg'], 
                             dataset_cfg['start_feature']:dataset_cfg['end_feature']]
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
        self.y_scaler = dataset_cfg.['y_scaler']
        if dataset_cfg['scale']==True:
            self.x_scaler.fit(train_data)
            if dataset_cfg['y_scale']==True:
                self.y_scaler.fit(input_data[dataset_cfg['train_start']:dataset_cfg['train_end'], dataset_cfg['target']].reshape(-1,1))
        if dataset_cfg['debutanizer']==False and dataset_cfg['scale']==True:
            gdb_data = self.x_scaler.transform(gdb_data)
            if dataset_cfg['y_scale']==True:
                gdb_label = self.y_scaler.transform(gdb_label.reshape(-1,1))
        # 选取有标签样本
        num_unlabel_per_interval = int(1/dataset_cfg['label_rate'])
        if num_unlabel_per_interval%2==0:
            left_idx = -int(num_unlabel_per_interval/2)+1
            right_idx = int(num_unlabel_per_interval/2)
        else:
            left_idx = -int((num_unlabel_per_interval-1)/2)
            right_idx = int((num_unlabel_per_interval-1)/2)
        labeled_data = []
        label = []
        for i in np.arange(gdb_data.shape[0]):
            if i%label_s == 0 and i>0 and i+right_idx+1<=gdb_data.shape[0]:
                labeled_data.append(gdb_data[i])
                label.append(gdb_label[i])
        self.data = torch.tensor(np.concatenate(labeled_data, axis=0).reshape(-1, gdb_data.shape[1]), dtype=torch.float32)
        self.label = torch.tensor(np.array(label, dtype=np.float), dtype=torch.float32)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return (self.data[idx],self.label[idx])



class TestLabelData(Dataset):
    def __init__(self, dataset_cfg:dict):
        input_data = np.genfromtxt(fname = dataset_cfg['root'] + dataset_cfg['name'], delimiter = ',')
        train_data = input_data[dataset_cfg['train_start']:dataset_cfg['train_end'], 
                                dataset_cfg['start_feature']:dataset_cfg['end_feature']]
        data_mean = train_data.mean(axis = 0).reshape(1, -1)
        data_var = train_data.var(axis = 0).reshape(1, -1)
        self.data = input_data[dataset_cfg['start_seg']:dataset_cfg['end_seg'], 
                              dataset_cfg['start_feature']:dataset_cfg['end_feature']]
        self.label = input_data[dataset_cfg['start_seg']:dataset_cfg['end_seg'], dataset_cfg['target']]
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
            self.data = input_data[dataset_cfg['start_seg']:dataset_cfg['end_seg']]
            self.label = input_label[dataset_cfg['start_seg']:dataset_cfg['end_seg']]
        # 变量标准化
        self.x_scaler = dataset_cfg['x_scaler']
        self.y_scaler = dataset_cfg['y_scaler']
        if dataset_cfg['scale']==True:
            self.x_scaler.fit(train_data)
            if dataset_cfg['y_scale']==True:
                self.y_scaler.fit(input_data[dataset_cfg['train_start']:dataset_cfg['train_end'], 
                                  dataset_cfg['target']].reshape(-1,1))
        if dataset_cfg['debutanizer']==False and dataset_cfg['scale']==True:
            self.data = self.x_scaler.transform(self.data)
            if dataset_cfg['y_scale']==True:
                self.label = self.y_scaler.transform(self.label.reshape(-1,1))
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        sample_data = torch.tensor(self.data[idx], dtype=torch.float32)
        sample_label = torch.tensor(self.label[idx], dtype=torch.float32)
        return sample_data, sample_label