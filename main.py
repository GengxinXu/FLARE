#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 20:49:57 2020

@author: xgx
"""

import torch
torch.cuda.set_device(0)
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn.functional as F
import os
import csv
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")
import random
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
import math

from networks import ReNet, E_lin1B, D_lin1, T, weights_init
from util import standardization, GetLoader, to_var, classification_accuracy, calc_TWcoef
from loss import kernel_gram_KLN, kernel_gram_z_KLN, cosine_similarity, prototype_triple_loss, reconstruction_loss
from sampling import imblearn_sampling, imblearn_sampling_Ran


#=============data=============
raw_data_root = "./Data/feature_total237_4sites.xlsx"
raw_data_feature_detail_root = "./Data/feature_group.xlsx"
fill_nan_for_hospital4 = True
delete_row_containing_nan = True
delete_row_with_label0 = True
transform_label = True # True or False # "1,2" -> "0", "3,4,5" -> "1"
data_preprocessing = 'standardization'
data_preprocessing_separately = 1
copy_target_data = True

#=============task=============
sep_data = 'semi_DA'
average_target = True # True or False
TTA = 0.3 # train_rate_for_all
if sep_data == 'semi_DA':
    idx_root = './Train_Test_idx_v5/'
    # label target
    train_idx_path_set = ['H1train1', 'H1train2', 'H1train3', 'H1train4', 'H1train5', 'H1train6', 'H1train7', 'H1train8', 'H1train9', 'H1train10'] # ['H1train1'], ['H2train1'], ['H3train1'], ['H4train1']
    # label source
    train_plus_idx_path_set = [[2],[2],[2], [2],[2],[2], [2],[2],[2], [2],[2],[2]] # [[2]], [[1]], [[1]], [[1]]
    # unlabel target
    test_idx_path_set = ['H1test1', 'H1test2', 'H1test3', 'H1test4', 'H1test5', 'H1test6', 'H1test7', 'H1test8', 'H1test9', 'H1test10'] # ['H1test1'], ['H2test1'], ['H3test1'], ['H4test1']
    idx_file_format = '.txt'
    train_rate_set = [TTA,TTA,TTA, TTA,TTA,TTA, TTA,TTA,TTA, TTA] # [TTA]
    train_idx_seed_set = [1,2,3,4,5,6,7,8,9,10] # [1]
        
#=============setting=============
ExpTime = 1 #
batch_size_sep = 100
n_classes = 2
SourceOnly_lr = 0.0002
Adam_beta1 = 0.5
Adam_beta2 = 0.999
expMethod=['FLARE']
overSample_type = 'SCBS'
largestMultiple=10 # None or a positive value
delta = 0.3 # the threshold reflects the frequency of updating class distribution in SCBS.
data_imblearn_ONsource_FORjointLearning = ''
data_imblearn_ONsource_FORtuningClassifier = 'RandomOverSampler'
data_imblearn_ONtarget_FORjointLearning = ''

#=============selectView=============
used_feat_group = ['all']# ['all'] or ['Gray features'] or ['Texture features'] or ['Histogram features'] or ['Number features'] or ['Intensity features'] or ['Surface features'] or ['Volume features'] or ........... ['Radiomic feature'] or ['Handcrafted feature'] 
if used_feat_group[0] == 'Radiomic feature':
    used_feat_group = ['Gray features','Texture features']
if used_feat_group[0] == 'Handcrafted feature':
    used_feat_group = ['Histogram features','Number features','Intensity features','Surface features','Volume features']


#=============import Data=============
raw_data = pd.read_excel(raw_data_root)
raw_data = raw_data.values
raw_data_feat = np.array(raw_data[:,6:], dtype=np.float32)
raw_data_label = np.array(raw_data[:,2], dtype=np.int32)
raw_data_hospital = np.array(raw_data[:,1], dtype=np.int32)
Ptext_conf_head = []
Ptext_conf_head.append("Raw_data has been loaded from <<{}>>, with shape {} and feature {} and label {}.".format(raw_data_root, raw_data.shape, raw_data_feat.shape, Counter(raw_data_label)))
print(Ptext_conf_head[-1])
try:
    raw_data_feature_detail = pd.read_excel(raw_data_feature_detail_root, sheetname='feature')
except TypeError:
    raw_data_feature_detail = pd.read_excel(raw_data_feature_detail_root, sheet_name='feature')
raw_data_feature_detail = raw_data_feature_detail.values
raw_data_feature_detail_view = np.array(raw_data_feature_detail[5:,2], dtype=np.str)
raw_data_feature_detail_view_group = []
for v_i in range(len(raw_data_feature_detail_view)):
    v_i = raw_data_feature_detail_view[v_i]
    if v_i not in raw_data_feature_detail_view_group:
        raw_data_feature_detail_view_group.append(v_i)
Ptext_conf_head.append("raw_data_feature_detail_view_group has been loaded from <<{}>>, with {} feature groups.".format(raw_data_feature_detail_root, len(raw_data_feature_detail_view_group)))
print(Ptext_conf_head[-1])
# ---- start filling for hos-4.
if fill_nan_for_hospital4:
    hos4_index = np.squeeze(np.argwhere(raw_data_hospital == 4))
    hosN4_index = np.squeeze(np.argwhere(raw_data_hospital != 4))
    raw_data_feat_hos4 = raw_data_feat[hos4_index,]
    raw_data_feat_hosN4 = raw_data_feat[hosN4_index,]
    raw_data_feat_hos4_fill = raw_data_feat_hos4
    raw_data_feat_hos4_nan = np.isnan(raw_data_feat_hos4)
    raw_data_feat_hos4_fill[raw_data_feat_hos4_nan] = 0
raw_data_feat = np.concatenate((raw_data_feat_hosN4,raw_data_feat_hos4_fill),axis=0)
# ---- end filling for hos-4.
if delete_row_containing_nan:
    raw_data_row_containing_nan = np.isnan(raw_data_feat).any(axis=1)
    raw_data_feat = raw_data_feat[~raw_data_row_containing_nan,]
    raw_data_label = raw_data_label[~raw_data_row_containing_nan,]
    raw_data_hospital = raw_data_hospital[~raw_data_row_containing_nan,]
    Ptext_conf_head.append("...... After delete_row_containing_nan, \n...... feature {} and label {}.".format(raw_data_feat.shape, Counter(raw_data_label)))
    print(Ptext_conf_head[-1])
if delete_row_with_label0:
    raw_data_row_containing_label0 = np.array(raw_data_label == 0, dtype=np.bool)#delete those row with label 0.
    raw_data_feat = raw_data_feat[~raw_data_row_containing_label0,]
    raw_data_label = raw_data_label[~raw_data_row_containing_label0,]
    raw_data_hospital = raw_data_hospital[~raw_data_row_containing_label0,]
    Ptext_conf_head.append("...... After delete_row_containing_label0, \n...... feature {} and label {}.".format(raw_data_feat.shape, Counter(raw_data_label)))
    print(Ptext_conf_head[-1])
if transform_label:
    raw_data_label = np.array(raw_data_label > 2, dtype=np.int64)
    Ptext_conf_head.append("...... After transform_label, \n...... feature {} and label {}".format(raw_data_feat.shape, Counter(raw_data_label)))
    print(Ptext_conf_head[-1])


if sep_data == 'semi_DA':
    num_tasks = len(train_idx_path_set)
for Domain_iter in range(num_tasks):
    Ptext = []
    Ptext_conf_add = []
    #-----------|Dataloder|-----------#
    source_domain_name = train_idx_path_set[Domain_iter]
    target_domain_name = test_idx_path_set[Domain_iter]
    train_idx_path = idx_root+source_domain_name+idx_file_format
    test_idx_path = idx_root+target_domain_name+idx_file_format
    if os.path.exists(train_idx_path) and os.path.exists(test_idx_path):
        train_idx = np.loadtxt(train_idx_path)
        train_idx = train_idx.astype(np.int32)
        test_idx = np.loadtxt(test_idx_path)
        test_idx = test_idx.astype(np.int32)
    else:
        source_hos_name = int(source_domain_name[1])
        used_index = np.squeeze(np.argwhere(raw_data_hospital == source_hos_name))
        used_index.sort()
        sample_n = len(used_index)
        train_rate = train_rate_set[Domain_iter]
        train_idx_seed = train_idx_seed_set[Domain_iter]
        if average_target:
            target_label = raw_data_label[used_index,]
            target_label_theLeastCount_value = min(Counter(target_label).values())
            target_label_theLeastCount_name  = list(Counter(target_label).keys())[np.squeeze(np.argwhere(np.array(list(Counter(target_label).values())) == target_label_theLeastCount_value))]
            target_label_theMostCount_value  = max(Counter(target_label).values())
            target_label_theMostCount_name   = list(Counter(target_label).keys())[np.squeeze(np.argwhere(np.array(list(Counter(target_label).values())) == target_label_theMostCount_value))]
            np.random.seed(train_idx_seed)
            test_idx_theLeastCount = np.random.choice(range(target_label_theLeastCount_value), int(target_label_theLeastCount_value*(1-train_rate)), replace=False)
            test_idx_theMostCount  = np.random.choice(range(target_label_theMostCount_value), int(target_label_theLeastCount_value*(1-train_rate)), replace=False)
            np.random.seed(None)
            test_idx_theLeastCount.sort()
            test_idx_theMostCount.sort()
            train_idx_theLeastCount = np.setdiff1d(range(target_label_theLeastCount_value),test_idx_theLeastCount)
            train_idx_theMostCount  = np.setdiff1d(range(target_label_theMostCount_value),test_idx_theMostCount)
            target_label_theLeastCount_idx = np.squeeze(np.argwhere(target_label == target_label_theLeastCount_name))
            target_label_theMostCount_idx  = np.squeeze(np.argwhere(target_label == target_label_theMostCount_name))
            test_idx_theLeastCount = target_label_theLeastCount_idx[test_idx_theLeastCount]
            test_idx_theMostCount  = target_label_theMostCount_idx[test_idx_theMostCount]
            train_idx_theLeastCount= target_label_theLeastCount_idx[train_idx_theLeastCount]
            train_idx_theMostCount = target_label_theMostCount_idx[train_idx_theMostCount]
            train_idx = np.concatenate([train_idx_theMostCount, train_idx_theLeastCount])
            test_idx  = np.concatenate([test_idx_theMostCount, test_idx_theLeastCount])
        else:
            np.random.seed(train_idx_seed)
            train_idx = np.random.choice(range(sample_n), int(sample_n*train_rate), replace=False)
            np.random.seed(None)
            train_idx.sort()
            test_idx = np.setdiff1d(range(sample_n),train_idx)
        train_idx = used_index[train_idx]
        test_idx = used_index[test_idx]
        if not os.path.exists(idx_root):
            os.makedirs(idx_root)
        np.savetxt(train_idx_path, train_idx)
        np.savetxt(test_idx_path, test_idx)
    Ptext_conf_add.append("\n>>>>>> hos-train(source) to hos-test(target) <<<<<<")
    print(Ptext_conf_add[-1])
    if sep_data == 'semi_DA':
        if data_preprocessing_separately == 1:
            source_feat_sep = raw_data_feat[train_idx,]
            target_feat_sep = raw_data_feat[test_idx,]
            if data_preprocessing == 'standardization':
                source_feat_sep, sourceMu, sourceSigma = standardization(source_feat_sep, axis=0)
                target_feat_sep, _, _ = standardization(target_feat_sep, axis=0, givenMu=sourceMu, givenSigma=sourceSigma)
                Ptext_conf_add.append("\nsource_feat_sep and target_feat_sep have been standarded \n")
                print(Ptext_conf_add[-1])
            else:
                Ptext_conf_add.append("\nsource_feat_sep and target_feat_sep are keeping raw data, without data_preprocessing \n")
                print(Ptext_conf_add[-1])
            if sep_data == 'semi_DA':
                source_feat_sep_Main = source_feat_sep
                source_feat_sep_notMain = source_feat_sep[[],:]
        if sep_data == 'semi_DA':
            train_idx_Main = train_idx
        num_of_idx_for_in_each_plus = []
        for source_plus_hos_name in train_plus_idx_path_set[Domain_iter]:
            train_plus_idx = np.squeeze(np.argwhere(raw_data_hospital == source_plus_hos_name))
            train_idx = np.concatenate([train_idx, train_plus_idx])
            source_domain_name = source_domain_name + 'H' + str(source_plus_hos_name)
            num_of_idx_for_in_each_plus.append(len(train_plus_idx))
            if data_preprocessing_separately == 1:
                source_feat_sep_add = raw_data_feat[train_plus_idx,]
                if data_preprocessing == 'standardization':
                    source_feat_sep_add, _, _ = standardization(source_feat_sep_add, axis=0)
                    Ptext_conf_add.append("\nsource_feat_sep_add has been standarded \n")
                    print(Ptext_conf_add[-1])
                else:
                    Ptext_conf_add.append("\nsource_feat_sep_add is keeping raw data, without data_preprocessing \n")
                    print(Ptext_conf_add[-1])
                source_feat_sep = np.concatenate([source_feat_sep,source_feat_sep_add])
                if sep_data == 'semi_DA':
                    source_feat_sep_notMain = np.concatenate([source_feat_sep_notMain,source_feat_sep_add])
    source_index = train_idx
    target_index = test_idx
    
    # sep data
    source_feat = raw_data_feat[source_index,]
    source_label = raw_data_label[source_index,]
    Ptext_conf_add.append("...... source_feat {} with label {}".format(source_feat.shape, Counter(source_label)))
    print(Ptext_conf_add[-1])
    target_feat = raw_data_feat[target_index,]
    target_label = raw_data_label[target_index,]
    Ptext_conf_add.append("...... target_feat {} with label {}".format(target_feat.shape, Counter(target_label)))
    print(Ptext_conf_add[-1])
    
    if data_preprocessing == 'standardization':
        source_feat, sourceMu, sourceSigma = standardization(source_feat,axis=0)
        target_feat, _, _ = standardization(target_feat,axis=0, givenMu=sourceMu, givenSigma=sourceSigma)
        Ptext_conf_add.append("\nsource_feat and target_feat have been standarded respectively \n")
        print(Ptext_conf_add[-1])
    else:
        Ptext_conf_add.append("\nsource_feat and target_feat are keeping raw data, without data_preprocessing \n")
        print(Ptext_conf_add[-1])
    if sep_data == 'semi_DA':
        if data_preprocessing_separately == 0:
            source_feat_notMain = source_feat[range(len(train_idx_Main),source_feat.shape[0]),:]
            source_feat = source_feat[range(0,len(train_idx_Main)),:]
        if data_preprocessing_separately == 1:
            if len(num_of_idx_for_in_each_plus) == 1:
                source_feat_notMain = source_feat_sep_notMain
            else:
                print("Undefined")
                exit()
            source_feat = source_feat_sep_Main
        if len(num_of_idx_for_in_each_plus) == 1:
            source_label_notMain = source_label[range(len(train_idx_Main),len(source_label))]
        else:
            print("Undefined")
            exit()
        source_label = source_label[range(0,len(train_idx_Main))]
        
        Ptext_conf_add.append("\nsource_feat is spited into source_feat(_Main) and source_feat_notMain, and target_feat is spited into target_feat(_Main) and target_feat_notMain, because of sep_data == 'semi_DA'")
        print(Ptext_conf_add[-1])
        Ptext_conf_add.append("...... source_feat(_Main) {} with label(_Main) {}".format(source_feat.shape, Counter(source_label)))
        print(Ptext_conf_add[-1])
        if len(num_of_idx_for_in_each_plus) == 1:
            Ptext_conf_add.append("...... source_feat_notMain {} with label_notMain {}".format(source_feat_notMain.shape, Counter(source_label_notMain)))
            print(Ptext_conf_add[-1])
        else:
            print("Undefined")
            exit()
        Ptext_conf_add.append("...... target_feat {} with label {}".format(target_feat.shape, Counter(target_label)))
        print(Ptext_conf_add[-1])
    # begin data_imblearn
    if data_imblearn_ONsource_FORjointLearning == '':
        Ptext_conf_add.append("\nsource_feat and source_label are keeping raw data, without data_imblearn \n")
        print(Ptext_conf_add[-1])
    else:
        source_feat, source_label = imblearn_sampling(feat=source_feat, label=source_label, data_imblearn_method=data_imblearn_ONsource_FORjointLearning)
        Ptext_conf_add.append("...... After data_imblear: '{}' for source_feat and source_label, \n...... feature {} and label {} in source_feat and source_label.".format(data_imblearn_ONsource_FORjointLearning, source_feat.shape, Counter(source_label)))
        print(Ptext_conf_add[-1])
        if sep_data == 'semi_DA':
            if len(num_of_idx_for_in_each_plus) == 1:
                source_feat_notMain, source_label_notMain = imblearn_sampling(feat=source_feat_notMain, label=source_label_notMain, data_imblearn_method=data_imblearn_ONsource_FORjointLearning)
                Ptext_conf_add.append("...... After data_imblear: '{}' for source_feat_notMain and source_label_notMain, \n...... feature {} and label {} in source_feat_notMain and source_label_notMain.".format(data_imblearn_ONsource_FORjointLearning, source_feat_notMain.shape, Counter(source_label_notMain)))
                print(Ptext_conf_add[-1])
            else:
                print("Undefined")
                exit()
    # end data_imblearn
    
    #generating DataLoader
    feat_some_view_idx = None
    if used_feat_group[0] == 'all':
        used_feat_group = raw_data_feature_detail_view_group
        input_dim = len(raw_data_feature_detail_view)
        feat_group_of_each_feat_Corresponding_to_X_with_some_feat = raw_data_feature_detail_view
    else:
        feat_some_view_idx = np.array([],dtype=np.int)
        for used_feat_group_i in range(len(used_feat_group)):
            feat_group_idx = np.squeeze(np.argwhere(raw_data_feature_detail_view == used_feat_group[used_feat_group_i]))
            feat_some_view_idx = np.concatenate((feat_some_view_idx,feat_group_idx),axis=0)
        input_dim = len(feat_some_view_idx)
        feat_group_of_each_feat_Corresponding_to_X_with_some_feat = raw_data_feature_detail_view[feat_some_view_idx]
        source_feat = source_feat[:,feat_some_view_idx]
        target_feat = target_feat[:,feat_some_view_idx]
    source_GetLoader = GetLoader(source_feat, source_label)
    target_GetLoader = GetLoader(target_feat, target_label)
    
    source_loader = DataLoader(source_GetLoader, batch_size=batch_size_sep, shuffle=True, drop_last=False, num_workers=0)
    test_loader = DataLoader(target_GetLoader, batch_size=batch_size_sep, shuffle=False, drop_last=False, num_workers=0)
    if target_feat.shape[0] > 0:
        target_loader = DataLoader(target_GetLoader, batch_size=batch_size_sep, shuffle=True, drop_last=False, num_workers=0)
    else:
        target_loader = test_loader
    #create target_sample_loader for CMMD loss
    target_feat_merge  = np.concatenate((source_feat, target_feat), axis=0)
    target_label_merge = np.concatenate((source_label,target_label),axis=0)
    target_merge_GetLoader = GetLoader(target_feat_merge, target_label_merge)
    target_sample_loader = DataLoader(target_merge_GetLoader, batch_size=batch_size_sep, shuffle=True, drop_last=False, num_workers=0)
    if copy_target_data and target_feat.shape[0] > 0:
        if target_feat.shape[0] < batch_size_sep:
            target_feat2 = np.concatenate((target_feat,target_feat),axis=0)
            target_label2 = np.concatenate((target_label,target_label),axis=0)
            while target_feat2.shape[0] < batch_size_sep:
                target_feat2 = np.concatenate((target_feat2,target_feat),axis=0)
                target_label2 = np.concatenate((target_label2,target_label),axis=0)
            target_GetLoader = GetLoader(target_feat2, target_label2)
            target_loader = DataLoader(target_GetLoader, batch_size=batch_size_sep, shuffle=True, drop_last=False, num_workers=0)
            #create target_sample_loader for CMMD loss
            target_feat2_merge  = np.concatenate((source_feat, target_feat2), axis=0)
            target_label2_merge = np.concatenate((source_label,target_label2),axis=0)
            target_merge2_GetLoader = GetLoader(target_feat2_merge, target_label2_merge)
            target_sample_loader = DataLoader(target_merge2_GetLoader, batch_size=batch_size_sep, shuffle=True, drop_last=False, num_workers=0)
    if sep_data == 'semi_DA':
        if len(num_of_idx_for_in_each_plus) == 1:
            if feat_some_view_idx is not None:
                source_feat_notMain = source_feat_notMain[:,feat_some_view_idx]
            source_GetLoader_notMain = GetLoader(source_feat_notMain, source_label_notMain)
            source_loader_notMain = DataLoader(source_GetLoader_notMain, batch_size=batch_size_sep, shuffle=True, drop_last=False, num_workers=0)
            if 1==1:
                source_sample_loader_notMain = DataLoader(source_GetLoader_notMain, batch_size=batch_size_sep, shuffle=True, drop_last=False, num_workers=0)
        else:
            print("Undefined")
            exit()
    
    Exp_iter = 0
    while Exp_iter < ExpTime:
        Exp_iter += 1
        print()
        if ('FLARE' in expMethod):
            total_epochs = 300
            # hyper-parameters
            cmmd_weight = 1
            multiViewRecLoss_startStep = 0
            multiViewRecLoss_weight = 0.002
            multiViewRecLoss_weight_declineAlpha = 0
            tripleLoss_startStep = 10
            tripleLoss_weight = 100
            tripleLoss_margin = 0
            tripleLoss_weight_declineAlpha = 0
            try:
                target_feat = target_feat2
                target_label = target_label2
            except NameError:
                pass
            e = E_lin1B(conv_dim = 64)
            d = D_lin1(conv_dim = 64, n_classes=n_classes)
            reNet = ReNet(in_dim=64, feat_group_of_each_feat=raw_data_feature_detail_view, feat_group=raw_data_feature_detail_view_group)
            t = T(inte_dim=512, out_dim=237)
            for i in [e,d,reNet,t]:
                i.cuda()
                i.apply(weights_init)
            SJ_solver = optim.Adam([{'params':e.parameters(),'lr':SourceOnly_lr*0.1}, {'params':d.parameters()}, {'params':reNet.parameters()}, {'params':t.parameters()}], SourceOnly_lr, [Adam_beta1, Adam_beta2], amsgrad=True)
            criterionQ_dis = nn.NLLLoss().cuda()
            MSELoss_func = nn.MSELoss().cuda()
            
            source_loader2 = source_loader# label target domain (1st batch from target domain for CMMD)
            target_sample_loader2 = target_sample_loader# whole target domain (including both label and unlabel) (2nd batch from target domain for CMMD)
            source_loader_notMain2 = source_loader_notMain# source domain (1st batch from source domain for CMMD)
            source_sample_loader_notMain2 = source_sample_loader_notMain# source domain (2nd batch from source domain for CMMD)
            sourceLab_selfInfo = np.ones(n_classes)
            targetUnl_selfInfo = np.ones(n_classes)
            sourceLab_sample_num = None
            targetUnl_sample_num = None
            Ptext.append("======> Target label domain: feature {} and label {}.".format(source_feat.shape, Counter(source_label)))
            print(Ptext[-1])
            Ptext.append("======> Source label domain:  feature {} and label {}.".format(source_feat_notMain.shape, Counter(source_label_notMain)))
            print(Ptext[-1])

            for step in range(total_epochs):
                print("\nstep: "+str(step))
                
                ############ Stochastic class-balanced boosting sampling (SCBS) phase ############
                
                epsilon_t = random.random()
                print("epsilon_t: {}".format(epsilon_t))
                if epsilon_t <= delta:
                    # 'SCBS':
                    if step > 0:
                        e.eval()
                        d.eval()
                        t.eval()
                        _,_,_,_,_,_,_,_, sourceLab_selfInfo = classification_accuracy(e, d, source_loader_notMain, domainTranslator=t, return_selfInfo=True)
                        _,_,_,_,_,_,_,_, targetUnl_selfInfo = classification_accuracy(e, d, source_loader, scalePrototype=True, return_selfInfo=True)
                    sourceLab_selfInfo_norm = [va/sum(sourceLab_selfInfo) for va in sourceLab_selfInfo]
                    targetUnl_selfInfo_norm = [va/sum(targetUnl_selfInfo) for va in targetUnl_selfInfo]
                    w_t = (1-np.cos((step+1)*np.pi/total_epochs))/2
                    sourceLab_selfInfo_NEW = (1-w_t)*np.ones(n_classes)/n_classes + (w_t)*np.array(sourceLab_selfInfo_norm, dtype=np.float64)
                    targetUnl_selfInfo_NEW = (1-w_t)*np.ones(n_classes)/n_classes + (w_t)*np.array(targetUnl_selfInfo_norm, dtype=np.float64)
                    # begin data_imblearn
                    if data_imblearn_ONsource_FORtuningClassifier == '':
                        source_loader2 = source_loader
                        target_sample_loader2 = target_sample_loader
                        Ptext.append("\nsource_feat and source_label are keeping raw data, without data_imblearn \n")
                        print(Ptext[-1])
                    else:
                        New_source_feat, New_source_label = imblearn_sampling_Ran(feat=source_feat, label=source_label, data_imblearn_method=data_imblearn_ONsource_FORtuningClassifier, defined_weight=targetUnl_selfInfo_NEW, sample_num=targetUnl_sample_num, largestMultiple=largestMultiple)
                        Ptext.append("...... After data_imblear: '{}' for tarLab_feat and tarLab_label, \n...... feature {} and label {} in source_feat and source_label.".format(data_imblearn_ONsource_FORtuningClassifier, New_source_feat.shape, Counter(New_source_label)))
                        print(Ptext[-1])
                        source_GetLoader2 = GetLoader(New_source_feat, New_source_label)
                        source_loader2 = DataLoader(source_GetLoader2, batch_size=batch_size_sep, shuffle=True, drop_last=False, num_workers=0)
                        target_feat_merge  = np.concatenate((New_source_feat, target_feat), axis=0)
                        target_label_merge = np.concatenate((New_source_label,target_label),axis=0)
                        target_merge_GetLoader = GetLoader(target_feat_merge, target_label_merge)
                        target_sample_loader2 = DataLoader(target_merge_GetLoader, batch_size=batch_size_sep, shuffle=True, drop_last=False, num_workers=0)
                    # end data_imblearn
                    # begin data_imblearn
                    if data_imblearn_ONsource_FORtuningClassifier == '':
                        source_loader_notMain2 = source_loader_notMain
                        source_sample_loader_notMain2 = source_sample_loader_notMain
                        Ptext.append("\nsource_feat and source_label are keeping raw data, without data_imblearn \n")
                        print(Ptext[-1])
                    else:
                        New_source_feat, New_source_label = imblearn_sampling_Ran(feat=source_feat_notMain, label=source_label_notMain, data_imblearn_method=data_imblearn_ONsource_FORtuningClassifier, defined_weight=sourceLab_selfInfo_NEW, sample_num=sourceLab_sample_num, largestMultiple=largestMultiple)
                        Ptext.append("...... After data_imblear: '{}' for source_feat and source_label, \n...... feature {} and label {} in source_feat and source_label.".format(data_imblearn_ONsource_FORtuningClassifier, New_source_feat.shape, Counter(New_source_label)))
                        print(Ptext[-1])
                        source_GetLoader_notMain2 = GetLoader(New_source_feat, New_source_label)
                        source_loader_notMain2 = DataLoader(source_GetLoader_notMain2, batch_size=batch_size_sep, shuffle=True, drop_last=False, num_workers=0)
                        source_sample_loader_notMain2 = DataLoader(source_GetLoader_notMain2, batch_size=batch_size_sep, shuffle=True, drop_last=False, num_workers=0)
                    # end data_imblearn
                else:
                    print("For the first epoch, use the original dataloder; for the latter epochs, remain the dataloder.")
                
                
                ############ representation learning phase ############
                
                Current_loss = []
                e.train()
                d.train()
                reNet.eval()
                t.train()
            
                for iteration, ((x_s, target_s), (_, _), (x_t, target_t), (_, _)) in enumerate(zip(source_loader2, source_loader_notMain2, target_sample_loader2, target_loader)):
                    if target_s.shape[0] == target_t.shape[0]:
                        onehot_s = torch.zeros(target_s.shape[0], n_classes).scatter_(1, target_s.view(-1,1), 1)
                        onehot_s = Variable(onehot_s).cuda()
                        x_s, target_s = Variable(x_s), Variable(target_s)
                        x_s, target_s = x_s.cuda(), target_s.cuda()
                        
                        onehot_t = torch.zeros(target_t.shape[0], n_classes).scatter_(1, target_t.view(-1,1), 1)
                        onehot_t = Variable(onehot_t).cuda()
                        x_t, target_t = Variable(x_t), Variable(target_t)
                        x_t, target_t = x_t.cuda(), target_t.cuda()
                        
                        e.zero_grad()
                        d.zero_grad()
                        reNet.zero_grad()
                        t.zero_grad()
                        
                        source_fe = e(x_s)
                        source_y = d(source_fe)
                        source_pre = F.softmax(source_y)
                        
                        target_fe = e(x_t)
                        target_y = d(target_fe)
                        target_pre = F.softmax(target_y)
                        
                        L_d = kernel_gram_KLN(onehot_s,onehot_s,target_s.shape[0],n_classes)
                        L_s = kernel_gram_KLN(target_pre,target_pre,target_t.shape[0],n_classes)
                        L_ds = kernel_gram_KLN(onehot_s,target_pre,target_s.shape[0],n_classes)
    
                        K_d = kernel_gram_z_KLN(source_fe,source_fe,target_s.shape[0],source_fe.shape[1])
                        K_s = kernel_gram_z_KLN(target_fe,target_fe,target_s.shape[0],source_fe.shape[1])
                        K_sd = kernel_gram_z_KLN(target_fe,source_fe,target_s.shape[0],source_fe.shape[1])
                        I = torch.eye(target_s.shape[0])
                        I = Variable(I).cuda()
                        lambda_ = 0.1
                        Inv_K_d = torch.inverse(K_d + lambda_*I)
                        Inv_K_s = torch.inverse(K_s + lambda_*I)
    
                        cmmd_t = (torch.trace(torch.mm(torch.mm(torch.mm(K_d, Inv_K_d), L_d), Inv_K_d)) +\
                                 torch.trace(torch.mm(torch.mm(torch.mm(K_s, Inv_K_s), L_s),Inv_K_s))- \
                                    2 * torch.trace(torch.mm(torch.mm(torch.mm(K_sd, Inv_K_d) ,L_ds ), Inv_K_s))) * 10
                        c_loss = cmmd_t * cmmd_weight
                        if step >= multiViewRecLoss_startStep:
                            reNet.train()
                            re_X_s = reNet(source_fe)
                            re_X_t = reNet(target_fe)
                            multiViewRecLoss_s = reconstruction_loss(re_X=re_X_s, raw_X=x_s, feat_group_of_each_feat=raw_data_feature_detail_view, feat_group=raw_data_feature_detail_view_group)
                            multiViewRecLoss_t = reconstruction_loss(re_X=re_X_t, raw_X=x_t, feat_group_of_each_feat=raw_data_feature_detail_view, feat_group=raw_data_feature_detail_view_group)
                            c_loss += (multiViewRecLoss_s + multiViewRecLoss_t) * multiViewRecLoss_weight
                        else:
                            multiViewRecLoss_s = torch.zeros(1)
                            multiViewRecLoss_t = torch.zeros(1)
                        c_loss.backward(retain_graph=True)
                        SJ_solver.step()
                        e.zero_grad()
                        d.zero_grad()
                        reNet.zero_grad()
                        t.zero_grad()
                        Current_loss.append(c_loss.item())
                        e.requires_grad=False
                        d.requires_grad=False
                        for _, ((x_s_notMain_A, target_s_notMain_A), (x_s_notMain_B, target_s_notMain_B)) in enumerate(zip(source_loader_notMain2, source_sample_loader_notMain2)):
                            onehot_s_notMain_A = torch.zeros(target_s_notMain_A.shape[0], n_classes).scatter_(1, target_s_notMain_A.view(-1,1), 1)
                            onehot_s_notMain_A = Variable(onehot_s_notMain_A).cuda()
                            x_s_notMain_A, target_s_notMain_A = Variable(x_s_notMain_A), Variable(target_s_notMain_A)
                            x_s_notMain_A, target_s_notMain_A = x_s_notMain_A.cuda(), target_s_notMain_A.cuda()

                            onehot_s_notMain_B = torch.zeros(target_s_notMain_B.shape[0], n_classes).scatter_(1, target_s_notMain_B.view(-1,1), 1)
                            onehot_s_notMain_B = Variable(onehot_s_notMain_B).cuda()
                            x_s_notMain_B, target_s_notMain_B = Variable(x_s_notMain_B), Variable(target_s_notMain_B)
                            x_s_notMain_B, target_s_notMain_B = x_s_notMain_B.cuda(), target_s_notMain_B.cuda()

                            source_notMain_A_tf = t(x_s_notMain_A)
                            source_notMain_A_fe = e(source_notMain_A_tf)
                            source_notMain_A_y = d(source_notMain_A_fe)
                            source_notMain_A_pre = F.softmax(source_notMain_A_y)

                            source_notMain_B_tf = t(x_s_notMain_B)
                            source_notMain_B_fe = e(source_notMain_B_tf)
                            source_notMain_B_y = d(source_notMain_B_fe)
                            source_notMain_B_pre = F.softmax(source_notMain_B_y)
                            
                            L_d = kernel_gram_KLN(onehot_s_notMain_A,onehot_s_notMain_A,target_s_notMain_A.shape[0],n_classes)
                            L_s = kernel_gram_KLN(source_notMain_B_pre,source_notMain_B_pre,target_s_notMain_B.shape[0],n_classes)
                            L_ds = kernel_gram_KLN(onehot_s_notMain_A,source_notMain_B_pre,target_s_notMain_A.shape[0],n_classes)
                            
                            K_d = kernel_gram_z_KLN(source_notMain_A_fe,source_notMain_A_fe,target_s_notMain_A.shape[0],source_notMain_A_fe.shape[1])
                            K_s = kernel_gram_z_KLN(source_notMain_B_fe,source_notMain_B_fe,target_s_notMain_B.shape[0],source_notMain_B_fe.shape[1])
                            K_sd = kernel_gram_z_KLN(source_notMain_B_fe,source_notMain_A_fe,target_s_notMain_A.shape[0],source_notMain_A_fe.shape[1])
                            I = torch.eye(target_s_notMain_A.shape[0])
                            I = Variable(I).cuda()
                            lambda_ = 0.1
                            Inv_K_d = torch.inverse(K_d + lambda_*I)
                            Inv_K_s = torch.inverse(K_s + lambda_*I)
                            
                            cmmd_s = (torch.trace(torch.mm(torch.mm(torch.mm(K_d, Inv_K_d), L_d), Inv_K_d)) +\
                                     torch.trace(torch.mm(torch.mm(torch.mm(K_s, Inv_K_s), L_s),Inv_K_s))- \
                                        2 * torch.trace(torch.mm(torch.mm(torch.mm(K_sd, Inv_K_d) ,L_ds ), Inv_K_s))) * 10
                            c_loss = cmmd_s * cmmd_weight
                            if step >= multiViewRecLoss_startStep:
                                reNet.train()
                                re_X_s_A = reNet(source_notMain_A_fe)
                                re_X_s_B = reNet(source_notMain_B_fe)
                                multiViewRecLoss_s_A = reconstruction_loss(re_X=re_X_s_A, raw_X=source_notMain_A_tf, feat_group_of_each_feat=raw_data_feature_detail_view, feat_group=raw_data_feature_detail_view_group)
                                multiViewRecLoss_s_B = reconstruction_loss(re_X=re_X_s_B, raw_X=source_notMain_B_tf, feat_group_of_each_feat=raw_data_feature_detail_view, feat_group=raw_data_feature_detail_view_group)
                                multiViewRecLoss_weight_NEW = multiViewRecLoss_weight * calc_TWcoef(step-multiViewRecLoss_startStep, multiViewRecLoss_weight_declineAlpha)
                                multiViewRecLoss_s = multiViewRecLoss_s_A + multiViewRecLoss_s_B
                                c_loss += multiViewRecLoss_s * multiViewRecLoss_weight_NEW
                            else:
                                multiViewRecLoss_weight_NEW = 0
                                multiViewRecLoss_s = torch.zeros(1)
                            if step >= tripleLoss_startStep:
                                target_mask = torch.zeros(source_notMain_A_pre.size()).scatter_(1, target_s_notMain_A.data.cpu().view(-1, 1), 1.)
                                target_mask = to_var(target_mask)
                                triple_loss_ite = prototype_triple_loss(tra_prototype2, source_notMain_A_fe, target_mask, tripleLoss_margin)
                                tripleLoss_weight_NEW = tripleLoss_weight * calc_TWcoef(step-tripleLoss_startStep, tripleLoss_weight_declineAlpha)
                                c_loss += triple_loss_ite * tripleLoss_weight_NEW
                            else:
                                tripleLoss_weight_NEW = 0
                                triple_loss_ite = torch.zeros(1)
                            c_loss.backward(retain_graph=True)
                            SJ_solver.step()
                            e.zero_grad()
                            d.zero_grad()
                            reNet.zero_grad()
                            t.zero_grad()
                            Current_loss.append(c_loss.item())
                
                e.eval()
                d.eval()
                t.eval()
                Current_loss = np.sum(Current_loss)/(len(Current_loss) - 1)
                tra_acc, tra_rec, tra_spe, tra_pre, tra_F1, tra_Gmean, _, _ = classification_accuracy(e, d, source_loader_notMain2, domainTranslator=t)
                val_acc, val_rec, val_spe, val_pre, val_F1, val_Gmean, _, _ = classification_accuracy(e, d, test_loader)
                _,_,_,_,_,_, tra_prototype2, _ = classification_accuracy(e, d, source_loader2, scalePrototype=True)
                setting = '[hos-{}]to[hos-{}]-Exp_{}: '.format(str(source_domain_name),str(target_domain_name),str(Exp_iter))
                loss_and_SourceResult = 'E:{} loss:{:0.4f} Sacc:{:0.4f} Srec:{:0.4f} '.format(step, Current_loss, tra_acc, tra_rec)
                TargetResult = 'Tacc:{:0.4f} Trec:{:0.4f} Tspe:{:0.4f} Tpre:{:0.4f} Tf1:{:0.4f} TGmean:{:0.4f}'.format(val_acc, val_rec, val_spe, val_pre, val_F1, val_Gmean)
                print(setting)
                print(loss_and_SourceResult)
                print(TargetResult)
                
        Record = 'resutls.txt'
        Record_conf = open(Record, "a")
        Record_conf.write(setting + loss_and_SourceResult + TargetResult + '\n')
        Record_conf.flush()
        
