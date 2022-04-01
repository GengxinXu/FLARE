#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 19:20:14 2020

@author: xgx
"""

import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

def standardization(data, axis=0, givenMu=None, givenSigma=None):
    if (givenMu is None) or (givenSigma is None):
        if axis != 0 and axis != 1:
            givenMu = np.nanmean(data)
            givenSigma = np.nanstd(data)
        else:
            givenMu = np.nanmean(data, axis=axis)
            givenSigma = np.nanstd(data, axis=axis)
    updateSigma = givenSigma
    try:
        updateSigma[updateSigma==0] = 1
    except TypeError:
        if updateSigma==0:
            updateSigma=1
    updateData = (data - givenMu) / updateSigma
    return updateData, givenMu, givenSigma

class GetLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    def __len__(self):
        return len(self.data)

def to_var(x):
    """Converts numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def classification_accuracy(e, d, test_loader, scalePrototype=False, domainTranslator=None, return_selfInfo=False):
    # if "scalePrototype=True", the prototype is calculated in the hypersphere
    correct = 0
    classnum = 2
    target_num = torch.zeros((1,classnum))
    predict_num = torch.zeros((1,classnum))
    acc_num = torch.zeros((1,classnum))
    all_pred = torch.empty(2)
    all_pred = to_var(all_pred)
    self_info = torch.zeros((1,classnum))
    self_info = to_var(self_info)
            
    for batch_idx, (X, target) in enumerate(test_loader):
        X, target = to_var(X), to_var(target).long().squeeze()
        test_fe = e(X)
        break
    prototype_matrix = torch.zeros((classnum,test_fe.size(-1)))
    prototype_matrix = to_var(prototype_matrix)
    
    for batch_idx, (X, target) in enumerate(test_loader):
        X, target = to_var(X), to_var(target).long().squeeze()
        if domainTranslator is not None:
            X = domainTranslator(X)
        test_fe = e(X)
        test_y = d(test_fe)
        output = F.softmax(test_y)

        pred = output.data.max(1)[1]
        all_pred = torch.cat([all_pred.long().squeeze(),pred])
        correct += pred.eq(target.data).cpu().sum()

        pre_mask = torch.zeros(output.size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
        predict_num += pre_mask.sum(0)
        tar_mask = torch.zeros(output.size()).scatter_(1, target.data.cpu().view(-1, 1), 1.)
        target_num += tar_mask.sum(0)
        acc_mask = pre_mask*tar_mask
        acc_num += acc_mask.sum(0)
        
        tar_mask = to_var(tar_mask)
        if scalePrototype:
            test_fe_length = (torch.sum(test_fe**2,1))**(1/2)
            test_fe_length = torch.t(test_fe_length.repeat(test_fe.size(-1),1))
            test_fe = test_fe / test_fe_length
        prototype_matrix = prototype_matrix + torch.mm(torch.t(tar_mask),test_fe) # prototype_matrix with size of 2 (class) x 64 (dim)
    
        theta1 = (output < (0.99999-1e-5)).float() * 1e-5
        theta2 = torch.mul( (output >= (0.99999-1e-5)).float(), 0.99999-output )
        self_info += (torch.mul(-torch.log(output+theta1+theta2), tar_mask)).sum(0)

    recall = (acc_num/target_num)[0][1]
    specificity = (acc_num/target_num)[0][0]
    precision = (acc_num/predict_num)[0][1] #may be nan, since maybe TP+FP=0
    F1 = 2*recall*precision/(recall+precision) #may be nan, since precision=nan
    Gmean = (recall*specificity) ** 0.5
    accuracy = correct.item()/len(test_loader.dataset)#or# accuracy = acc_num.sum(1)/target_num.sum(1)
    pseudo_label = all_pred.cpu().numpy()[2:] # delete the 0 we assign when defining all_pred
    
    self_info = (self_info.cpu()/target_num)[0]
    
    prototype_matrix = prototype_matrix / to_var(torch.t((target_num[0]).repeat(prototype_matrix.size(-1),1))) # calculate the mean for each class

    if return_selfInfo:
        return accuracy, recall.item(), specificity.item(), precision.item(), F1.item(), Gmean.item(), prototype_matrix, pseudo_label, self_info.detach().numpy()
    else:
        return accuracy, recall.item(), specificity.item(), precision.item(), F1.item(), Gmean.item(), prototype_matrix, pseudo_label

def calc_TWcoef(step_num, alpha=0):
    return 1-np.float((1-np.exp(-alpha*step_num))/(1+np.exp(-alpha*step_num)))
