#!/usr/bin/env python
# coding: utf-8
import os
import ccn_io
import ccn_preprocess
import ccn_algorithms
import ccn_ml
import ccn_visualization
import numpy as np
import ccn_wo_prior_main as cc
accs   = []
tmrngs = []

# get the subject you want
#subjs = [x[0] for x in os.walk("data/data_in/")]
subjs = ["subj01","subj02","subj03"]
# get the information you need
for subj in subjs:
    print("---------subj_change-----------")
    folder = "../../Data/"+subj +"/"+subj+"_prior_video"
    print(folder)
    fileList = [folder + "_android.mat",folder + "_human.mat",folder + "_robot.mat"]
    accuracies, time_ranges = cc.generalized_pipeline(fileList,window_size = 100, window_shift = 100 ,verbose = False,plot = True)
    print(accuracies)
    accs.append(accuracies)
    tmrngs.append(time_ranges)

for i in range(len(accs)):
    print(accs[i])
    print(tmrngs[i])
