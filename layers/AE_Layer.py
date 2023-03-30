# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class AutoEncoder(nn.Module):
    def __init__(self, dim_X, dim_H, act):
        super(AutoEncoder, self).__init__()
        self.dim_X = dim_X
        self.dim_H = dim_H
        activation_func = {
            "sigmoid":torch.sigmoid,
            "tanh":torch.tanh,
            "relu":F.relu,
            "gelu":F.gelu,
            "leaky relu":F.leaky_relu,
            "None":None
        }
        self.act = activation_func.get(act, None)

        self.encoder = nn.Linear(dim_X, dim_H, bias=True)
        self.decoder = nn.Linear(dim_H, dim_X, bias=True)

    def forward(self, X, rep=False):

        if self.act is not None:
            H = self.act(self.encoder(X))
            if rep is False:
                return self.act(self.decoder(H))
            else:
                return H
        else:
            H = self.encoder(X)
            if rep is False:
                return self.decoder(H)
            else:
                return H



class QualityAutoEncoder(nn.Module):
    def __init__(self, dim_X, dim_H, act, act_reg):
        super(QualityAutoEncoder, self).__init__()
        self.dim_X = dim_X
        self.dim_H = dim_H
        activation_func = {
            "sigmoid":torch.sigmoid,
            "tanh":torch.tanh,
            "relu":F.relu,
            "gelu":F.gelu,
            "leaky relu":F.leaky_relu,
            "None":None
        }
        self.act_decode = activation_func.get(act, None)
        self.act_reg = activation_func.get(act_reg, None)

        self.encoder = nn.Linear(dim_X, dim_H, bias=True)
        self.decoder = nn.Linear(dim_H, dim_X, bias=True)
        self.regressor = nn.Linear(dim_H, 1, bias=True)

    def forward(self, X, rep=False):

        if self.act_decode is not None and self.act_reg is not None:
            H = self.act_decode(self.encoder(X))
            if rep is False:
                return torch.cat((self.act_decode(self.decoder(H)),self.act_reg(self.regressor(H))), 1)
            else:
                return H
        elif self.act_decode is not None:
            H = self.act_decode(self.encoder(X))
            if rep is False:
                return torch.cat((self.act_decode(self.decoder(H)),self.regressor(H)), 1)
            else:
                return H
        elif self.act_reg is not None:
            H = self.self.encoder(X)
            if rep is False:
                return torch.cat((self.decoder(H),self.act_reg(self.regressor(H))), 1)
            else:
                return H
        else:
            H = self.self.encoder(X)
            if rep is False:
                return torch.cat((self.decoder(H),self.regressor(H)), 1)
            else:
                return H