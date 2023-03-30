# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from AE_Layer import AutoEncoder, QualityAutoEncoder



class StackedAutoEncoder(nn.Module):
    def __init__(self, cfg, device=torch.device('cpu')):
        super(StackedAutoEncoder, self).__init__()
        self.AElength = len(cfg.SAE_size)
        SAE = []
        self.device = device

        for i in range(1, self.AElength):
            SAE.append(AutoEncoder(cfg.SAE_size[i-1], cfg.SAE_size[i], cfg.act_SAEencdec).to(device))
        self.SAE = nn.ModuleList(SAE)

    def forward(self, X, NoL, PreTrain=False):
        """
        :param X: 进口参数
        :param NoL: 第几层
        :param PreTrain: 是不是无监督预训练
        :return:
        """
        out = X
        if PreTrain is True:
            # SAE的预训练
            if NoL == 0:
                return out, self.SAE[NoL](out)

            else:
                for i in range(NoL):
                    # 第N层之前的参数给冻住
                    for param in self.SAE[i].parameters():
                        param.requires_grad = False

                    out = self.SAE[i](out, rep=True)
                # 训练第N层
                inputs = out
                out = self.SAE[NoL](out)
                return inputs, out
        else:
            for i in range(self.AElength-1):
                # 做微调，输出的是最后一层隐变量
                for param in self.SAE[i].parameters():
                    param.requires_grad = True

                out = self.SAE[i](out, rep=True)
            return out



class StackedQualityAutoEncoder(nn.Module):
    def __init__(self, cfg, device=torch.device('cpu')):
        super(StackedQualityAutoEncoder, self).__init__()
        self.AElength = len(cfg.SAE_size)
        SAE = []
        self.device = device

        for i in range(1, self.AElength):
            SAE.append(QualityAutoEncoder(cfg.SAE_size[i-1], cfg.SAE_size[i], cfg.act_SAEencdec, cfg.act_SAEreg).to(device))
        self.SAE = nn.ModuleList(SAE)
        
    def forward(self, X, NoL, PreTrain=False):
        """
        :param X: 进口参数
        :param NoL: 第几层
        :param PreTrain: 是不是无监督预训练
        :return:
        """
        out = X
        if PreTrain is True:
            # SAE的预训练
            if NoL == 0:
                return out, self.SAE[NoL](out)

            else:
                for i in range(NoL):
                    # 第N层之前的参数给冻住
                    for param in self.SAE[i].parameters():
                        param.requires_grad = False

                    out = self.SAE[i](out, rep=True)
                # 训练第N层
                inputs = out
                out = self.SAE[NoL](out)
                return inputs, out
        else:
            for i in range(self.AElength-1):
                # 做微调，输出的是最后一层隐变量
                for param in self.SAE[i].parameters():
                    param.requires_grad = True

                out = self.SAE[i](out, rep=True)
            return out



class SAEConfig:
    def __init__(self):
        self.labeled_data_step = 1
        # ------------------------data config------------------------ #
        # fault_list and fault_dict are parameters for classification tasks
        # train_seg, valid_seg, test_seg are parameters for regression tasks
        self.train_dataset = dict(
            name = "gaodibian_140000_160000.CSV",
            root = "E:\\SS_SAE_graph_regularization\Data\\",
            scale = True,
            y_scale = False,
        )
        self.validate_dataset = dict(
            name = "gaodibian_140000_160000.CSV",
            root = "E:\\SS_SAE_graph_regularization\Data\\",
            scale = True,
            y_scale = False,
        )
        self.test_dataset = dict(
            name = "gaodibian_140000_160000.CSV",
            root = "E:\\SS_SAE_graph_regularization\Data\\",
            scale = True,
            y_scale = False,
        )
        # ------------------Stacked AE config---------------------- #
        self.SAE_size = [13,7,5,1]
        self.act_SAEencdec="tanh"
        self.act_SAEreg="None"