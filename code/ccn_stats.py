#!/usr/bin/env python
# coding: utf-8

# ## Use t-test
# -----------------
# - Regression, logistic regression, cluster analysis, statistical graphics, quantile regression
# - compare two means or propotions
# 
# 1. Unmatched t-test: independent samples
# 2. Matched t-test: dependency
# 
# ### Assumptions
# ----------
# 1. Variances of two populations are equal 
# 2. T-test makes sense only if means make sense
# 
# ### Alternatives
# -----------
# 1. Wilcoxon test
# 2. Permutation test
# 3. Bootstrap
# 

# Import needed libraries and data

import glob
import json
import os

import ccn_visualization
import numpy  as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_ind
from statsmodels.sandbox.stats.multicomp import multipletests


def read_and_prepare_data(file):
    # read data
    with open(file) as f:
        data = json.load(f)

    avg_values = [[k, v] for k, v in data['avg_all'].items()]
    del (data['avg_all'])

    # keys for json object
    subjects = list(data.keys())
    windows = list(data[subjects[0]].keys())

    # information storage
    eeg_sliced = {}
    for each in windows:
        eeg_sliced[each] = []  # initialization with empty list

    # store info
    for window in windows:
        for subject in subjects:
            eeg_sliced[window].append(data[subject][window])

    return avg_values, eeg_sliced


def compare_with_chance_level(eeg_sliced):

    n_subjects = len(next(iter(eeg_sliced.values())))
    chance_level = {key: [0.3333]*n_subjects for key in eeg_sliced.keys()}

    return compare_vectors(eeg_sliced, chance_level)


def compare_vectors(vector1, vector2):
    '''Find significantly different time windows'''

    windows = list(vector1.keys())

    ttest_values = []
    for window in windows:
        ttest_values.append(ttest_ind(vector1[window], vector2[window]))

        # combine info
    t_values = np.asarray(ttest_values)
    p_values = [float(i) for i in t_values[:, -1]]

    # correction
    p_adjusted = multipletests(p_values, method='bonferroni')[:2]
    p_adjusted_l = list(map(list, zip(*[x.tolist() for x in p_adjusted])))

    # significant indices
    sig_indices = np.where(p_adjusted[0])[0].tolist()

    return sig_indices

def choose_from_options( list_options, replace_option = ""):
    """Printing options the easily for the user to use."""
    
    separator = "\n-------------------------------------------------------\n"
    
    for index in range(len(list_options)):
        print("["+str(index+1)+"] "+ list_options[index].replace( replace_option,""))
    
    choice  = int(input('Enter choice: '))
    print("\n")
    
    return choice-1

def choose( folder_list):
    """Choose the accuracies you want to work with."""

    # get directory you want to work with
    folder_num  = choose_from_options( folder_list)
    
    # get the file you want to work with
    folders =  glob.glob( folder_list[ folder_num] + "*/accuracy_results.json")
    subdir = choose_from_options( folders, folder_list[folder_num])
    
    # choose
    print("You chose this file for analysis: " + folders[subdir]+ ' .')
    response = input('Do you want to continue?(Y/anything else)')
    
    if response not in ['Y','y']:
        return
    
    else:
        avg_values, eeg_sliced = read_and_prepare_data(folders[subdir])
        sig_index = compare_with_chance_level(eeg_sliced)

        windows = list(eeg_sliced.keys())
        dir = os.path.dirname(folders[subdir])  ## directory of file
        ccn_visualization.visualize(dir, '/avg_accuracy.png', sig_index, windows, avg_values)


def overlay(video_path, still_path):
    "Takes paths to results json of video and still input experiments, and overlays the significance analysis"

    v_avg_vals, v_eeg_sliced = read_and_prepare_data(video_path)
    v_sig_index = compare_with_chance_level(v_eeg_sliced)

    s_avg_vals, s_eeg_sliced = read_and_prepare_data(still_path)
    s_sig_index = compare_with_chance_level(s_eeg_sliced)

    windows = list(v_eeg_sliced.keys())

    if(windows != list(s_eeg_sliced.keys())):
        raise Exception('Still and video window values are not equal')

    dir = os.path.dirname(video_path)  ## directory of file
    ccn_visualization.visualize_still_and_video(dir,'/overlayed_avg_accuracy.png', v_sig_index, s_sig_index, windows, v_avg_vals, s_avg_vals)

    # TODO: Ask what is the folder list and when it is necessary

def run_all(folder_list, overlaying=False):

    # Run all the information that is in each folder.
    for each in folder_list:
        folders = glob.glob(each + "*/accuracy_results.json")
        for each_subdir in folders:
            #print(each_subdir)
            avg_values, eeg_sliced = read_and_prepare_data(each_subdir)
            sig_index = compare_with_chance_level(eeg_sliced)

            windows = list(eeg_sliced.keys())
            dir = os.path.dirname(each_subdir)  ## directory of file
            ccn_visualization.visualize(dir, '/avg_accuracy.png', sig_index, windows, avg_values)

def overlay_all(main_folder_path):
    # name of folders to take the information from
    folders = [name for name in os.listdir(main_folder_path) if os.path.isdir(main_folder_path + name)]
    for folder in folders:
        # create the file path
        file_pth = main_folder_path + folder + '/'
        # get all the files
        files = [file_pth+name for name in os.listdir(file_pth) if name.endswith(".json")]
        if files:
            overlay(files[0],files[1])
        else:
            print("Error: No files were found")

if __name__ == "__main__":
    overlay_all("/home/sena/Desktop/Experiments/")