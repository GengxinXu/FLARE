#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 19:50:36 2020

@author: xgx
"""

import torch
import numpy as np
import torch.nn as nn
import random
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
import math

def imblearn_sampling(feat, label, data_imblearn_method, random_state=0):
    if data_imblearn_method == 'RandomOverSampler':
        try:
            ros = RandomOverSampler(random_state=random_state)
            try:
                feat, label = ros.fit_sample(feat, label)
            except AttributeError:
                feat, label = ros.fit_resample(feat, label)
        except NameError:
                feat, label = imblearn_sampling_Ran(feat=feat, label=label, data_imblearn_method='RandomOverSampler', defined_weight=np.ones(n_classes)/n_classes)
    feat = np.float32(feat)
    return feat, label

def sortedDictValues(adict,reverse=False):
    keys = list(adict.keys())
    keys.sort(reverse=reverse)
    return keys, [adict[key] for key in keys]

def self_RandomOverSampler_1(feat, label, weight, largestMultiple):
    if largestMultiple is not None:
        if (max(weight) / min(weight)) > largestMultiple:
            for i_class in range(len(weight)):
                if weight[i_class] > (min(weight)*largestMultiple):
                    weight[i_class] = min(weight)*largestMultiple
    new_feat = feat
    new_label = label
    class_sortByClass, sample_sortByClass = sortedDictValues(Counter(label))
    sample_sortByClass_pab = [va/sum(sample_sortByClass) for va in sample_sortByClass]
    if np.isnan(weight[0]):
        weight2 = [1/len(weight) for va in range(len(weight))]
        weight = weight2
    weight_pab = [va/sum(weight) for va in weight]
    diff_pab = [sample_sortByClass_pab[va]-weight_pab[va] for va in range(len(weight))]
    fix_class_ind = np.squeeze(np.argwhere(np.array(diff_pab) == max(diff_pab)))
    fix_class_sample = sample_sortByClass[fix_class_ind]
    final_sample_N_for_each = [math.ceil(va / weight[fix_class_ind] * fix_class_sample) for va in weight]
    for i_class in range(len(class_sortByClass)):
        i_class_name = class_sortByClass[i_class]
        i_class_raw_sample = sample_sortByClass[i_class]
        i_class_final_sample = final_sample_N_for_each[i_class]
        i_class_adding_sample = i_class_final_sample - i_class_raw_sample
        if i_class_adding_sample > 0:
            used_index = np.squeeze(np.argwhere(label == i_class_name))
            overSample_index = np.random.choice(used_index, i_class_adding_sample, replace=True)
            add_feat = feat[overSample_index,:]
            add_label = label[overSample_index]
            add_label = np.array(add_label, dtype=np.int64)
            new_feat = np.concatenate([new_feat, add_feat])
            new_label = np.concatenate([new_label, add_label])
    return new_feat, new_label

def self_RandomOverSampler_2(feat, label, sample_num):
    new_feat = feat
    new_label = label
    class_sortByClass, sample_sortByClass = sortedDictValues(Counter(label))
    for i_class in range(len(class_sortByClass)):
        i_class_name = class_sortByClass[i_class]
        i_class_raw_sample = sample_sortByClass[i_class]
        i_class_final_sample = math.ceil(sample_num[i_class])
        i_class_adding_sample = i_class_final_sample - i_class_raw_sample
        if i_class_adding_sample >= 0:
            used_index = np.squeeze(np.argwhere(label == i_class_name))
            raw_feat = feat[used_index ,:]
            raw_label = label[used_index ]
            raw_label = np.array(raw_label, dtype=np.int64)
            new_feat = np.concatenate([new_feat, raw_feat])
            new_label = np.concatenate([new_label, raw_label])
            if i_class_adding_sample > 0:
                overSample_index = np.random.choice(used_index, i_class_adding_sample, replace=True)
                add_feat = feat[overSample_index,:]
                add_label = label[overSample_index]
                add_label = np.array(add_label, dtype=np.int64)
                new_feat = np.concatenate([new_feat, add_feat])
                new_label = np.concatenate([new_label, add_label])
        else:# i_class_adding_sample < 0:
            used_index = np.squeeze(np.argwhere(label == i_class_name))
            selectedSample_index = np.random.choice(used_index, i_class_final_sample, replace=False)
            add_feat = feat[selectedSample_index,:]
            add_label = label[selectedSample_index]
            add_label = np.array(add_label, dtype=np.int64)
            new_feat = np.concatenate([new_feat, add_feat])
            new_label = np.concatenate([new_label, add_label])
    new_feat = new_feat[feat.shape[0]:new_feat.shape[0],:]
    new_label = new_label[label.shape[0]:new_label.shape[0]]
    return new_feat, new_label

def imblearn_sampling_Ran(feat, label, data_imblearn_method, defined_weight=None, sample_num=None, largestMultiple=None):
    if (defined_weight is None) and (sample_num is None):
        if data_imblearn_method == 'RandomOverSampler':
            try:
                ros = RandomOverSampler()
                try:
                    feat, label = ros.fit_sample(feat, label)
                except AttributeError:
                    feat, label = ros.fit_resample(feat, label)
            except NameError:
                    feat, label = self_RandomOverSampler_1(feat=feat, label=label, weight=np.ones(n_classes)/n_classes, largestMultiple=largestMultiple)
    else:
        if (defined_weight is not None) and (data_imblearn_method == 'RandomOverSampler'):
            feat, label = self_RandomOverSampler_1(feat, label, defined_weight, largestMultiple)
        if (sample_num is not None) and (data_imblearn_method == 'RandomOverSampler'):
            feat, label = self_RandomOverSampler_2(feat, label, sample_num)
    feat = np.float32(feat)
    return feat, label

