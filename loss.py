#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 19:23:07 2020

@author: xgx
"""

import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

def kernel_gram_KLN(x,y,n,d):
    zx = x.repeat(1,n).resize(n*n,d)
    zy = y.repeat(n,1)
    z = torch.pow(torch.nn.functional.pairwise_distance(zx,zy),2)
    zz = z.resize(n,n)
    res = torch.zeros(n,n)
    res = Variable(res).cuda()
    bw=1
    res = torch.exp(-zz/(2*bw))
    return res  

def kernel_gram_z_KLN(x,y,n,d):
    mn = 5
    #bw = [10]
    bw = [1,5,10,20,40]
    zx = x.repeat(1,n).resize(n*n,d)
    zy = y.repeat(n,1)
    z = torch.pow(torch.nn.functional.pairwise_distance(zx,zy),2)
    zz = z.resize(n,n)
    res = torch.zeros(n,n)
    res = Variable(res).cuda()
    for i in range(mn):
        res = res + torch.exp(-zz/(2*bw[i]))
    return res/5   

def cosine_similarity(x,y,n,m,d):
    zx = x.repeat(1,m).resize(n*m,d)
    zy = y.repeat(n,1)
    z = torch.sum(torch.mul(zx,zy),1)
    zz = z.resize(n,m)
    xx_len = torch.sum(torch.mul(x,x),1).sqrt()
    xx_len = torch.t(xx_len.repeat(m,1))
    yy_len = torch.sum(torch.mul(y,y),1).sqrt()
    yy_len = yy_len.repeat(n,1)
    zz = zz / xx_len
    zz = zz / yy_len
    return zz

def prototype_triple_loss(prototype, feature, label_mask, margin=0):
    cos_Matrix = cosine_similarity(prototype, feature, prototype.size(0), feature.size(0), prototype.size(1))
    positive_Matrix = torch.mul(cos_Matrix,torch.t(label_mask)) 
    negative_Matrix = torch.mul(cos_Matrix,torch.t(1-label_mask))
    pos_min = torch.min(positive_Matrix + torch.t(1-label_mask),1)[0]
    neg_max = torch.max(negative_Matrix - torch.t(label_mask),1)[0]
    mean_dis = torch.mean(pos_min - neg_max)
    gap = min(mean_dis - margin, 0*mean_dis)
    return -gap

def reconstruction_loss(re_X, raw_X, feat_group_of_each_feat=None, feat_group=None, used_feat_group=None, wholeDistance=False):
    if wholeDistance:
        loss = torch.pow(F.pairwise_distance(raw_X, re_X, p=2), 2).sum()
    else:
        if (feat_group_of_each_feat is None) or (feat_group is None):
            print('ERROR: "feat_group_of_each_feat" and "feat_group" are required in "reconstruction_loss".')
            exit()
        if used_feat_group is None:
            used_feat_group = feat_group
        loss = 0
        for feat_group_i in range(len(feat_group)):
            if feat_group[feat_group_i] in used_feat_group:
                feat_group_idx = np.squeeze(np.argwhere(feat_group_of_each_feat == feat_group[feat_group_i]))
                len_of_feat = feat_group_idx.shape[0]
                loss = loss + torch.pow(F.pairwise_distance(raw_X[:,feat_group_idx], re_X[str(feat_group_i)], p=2), 2).sum()/len_of_feat
    return loss/raw_X.size(0)
