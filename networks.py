#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 19:01:02 2020

@author: xgx
"""

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

# Reconstruction network
class ReNet(nn.Module):
    def __init__(self, in_dim, feat_group_of_each_feat, feat_group):
        super(ReNet, self).__init__()
        self.lin0 = nn.Linear(in_dim, len(np.squeeze(np.argwhere(feat_group_of_each_feat == feat_group[0]))))
        self.lin1 = nn.Linear(in_dim, len(np.squeeze(np.argwhere(feat_group_of_each_feat == feat_group[1]))))
        self.lin2 = nn.Linear(in_dim, len(np.squeeze(np.argwhere(feat_group_of_each_feat == feat_group[2]))))
        self.lin3 = nn.Linear(in_dim, len(np.squeeze(np.argwhere(feat_group_of_each_feat == feat_group[3]))))
        self.lin4 = nn.Linear(in_dim, len(np.squeeze(np.argwhere(feat_group_of_each_feat == feat_group[4]))))
        self.lin5 = nn.Linear(in_dim, len(np.squeeze(np.argwhere(feat_group_of_each_feat == feat_group[5]))))
        self.lin6 = nn.Linear(in_dim, len(np.squeeze(np.argwhere(feat_group_of_each_feat == feat_group[6]))))
    def forward(self, x):
        x0 = self.lin0(x)
        x1 = self.lin1(x)
        x2 = self.lin2(x)
        x3 = self.lin3(x)
        x4 = self.lin4(x)
        x5 = self.lin5(x)
        x6 = self.lin6(x)
        return {'0':x0, '1':x1, '2':x2, '3':x3, '4':x4, '5':x5, '6':x6}

# Encoder
class E_lin1B(nn.Module):
    def __init__(self, conv_dim=64, input_dim=237):
        super(E_lin1B, self).__init__()
        self.lin1 = nn.Linear(input_dim, conv_dim)
    def forward(self, x):
        x = self.lin1(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        return x

# Decoder
class D_lin1(nn.Module):
    def __init__(self, conv_dim=64, n_classes=None):
        super(D_lin1, self).__init__()
        self.lin1 = nn.Linear(conv_dim, n_classes)
    def forward(self, x):
        return self.lin1(x)

# Transformation layer
class T(nn.Module):
    def __init__(self, inte_dim=512, out_dim=237, input_dim=237):
        super(T, self).__init__()
        self.lin1 = nn.Linear(input_dim, inte_dim)
        self.lin2 = nn.Linear(inte_dim, out_dim)
    def forward(self, x):
        x = self.lin1(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.lin2(x)
        return x

# init
def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            dm.weight.data.normal_(0.0, 0.05)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.05)
            m.bias.data.fill_(0) 
