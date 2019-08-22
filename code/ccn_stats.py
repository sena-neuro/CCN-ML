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

import ccn_visualization
import glob
import json
import os

import numpy  as np
import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.sandbox.stats.multicomp import multipletests


def read_and_prepare_data(file):
    # read data
    with open(file) as f:
        data = json.load(f)

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

    avg_values = [[k, v] for k, v in data['avg_all'].items()]

    return avg_values, eeg_sliced


def analyze(avg_values, eeg_sliced):
    """Extract significant information from the json file."""

    windows = list(eeg_sliced.keys())

    ttest_values = []
    for window in windows:
        ttest_values.append(ttest_1samp(eeg_sliced[window], 0.333))  # Why [:-1] and not all of eeg_sliced

    # combine info
    # avg_values = [[k, v] for k, v in data['avg_all'].items()]
    t_values = np.asarray(ttest_values)
    p_values = [float(i) for i in t_values[:, -1]]

    # correction
    p_adjusted = multipletests(p_values, method='bonferroni')[:2]
    p_adjusted_l = list(map(list, zip(*[x.tolist() for x in p_adjusted])))

    # for ease of use - pandas DataFrame
    values = np.concatenate((avg_values, t_values, p_adjusted_l), axis=1)
    data = pd.DataFrame(values)  # pandas
    data.columns = ['window', 'accuracy', 't-statistic', 'pvalue', 'truthfulness', 'new_p']

    # decide on time
    windows_val = [2 * (x - 100) for x in [int(wind_frame.strip('()').split(',')[0])
                                           for wind_frame in windows]]

    # average values
    vals = [x[-1] for x in avg_values][:len(windows_val)]

    # p significance
    sig_index = data[data['truthfulness'] == '1.0']['accuracy'].index.tolist()[:len(windows_val)]

    return sig_index, windows_val, vals




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
        avg_vals, eeg_sliced = read_and_prepare_data(folders[subdir])
        s, w, v = analyze(avg_vals, eeg_sliced)
        dir = os.path.dirname(folders[subdir])  ## directory of file
        ccn_visualization.visualize(dir, s, w, v)


def overlaying_analysis(video_path, still_path):
    "Takes paths to results json of video and still input experiments, and overlays the significance analysis"

    v_avg_vals, v_eeg_sliced = read_and_prepare_data(video_path)
    v_sig_index, v_windows_val, v_vals = analyze(v_avg_vals, v_eeg_sliced)

    s_avg_vals, s_eeg_sliced = read_and_prepare_data(still_path)
    s_sig_index, s_windows_val, s_vals = analyze(s_avg_vals, s_eeg_sliced)

    if (v_windows_val == s_windows_val):
        windows_val = s_windows_val
    else:
        raise Exception('Still and video window values are not equal')

    return v_sig_index, s_sig_index, windows_val, v_vals, s_vals

    # TODO: Ask what is the folder list and when it is necessary


def run_all(folder_list, overlaying=False):

    # Run all the information that is in each folder.
    for each in folder_list:
        folders = glob.glob(each + "*/accuracy_results.json")
        for each_subdir in folders:
            #print(each_subdir)
            avg_values, info_storage = read_and_prepare_data(each_subdir)
            s, w, v = analyze(avg_values, info_storage)
            dir = os.path.dirname(each_subdir)  ## directory of file
            ccn_visualization.visualize(dir, s, w, v)

def overlay_all(main_folder_path):
    # name of folders to take the information from
    folders = [name for name in os.listdir(main_folder_path) if os.path.isdir(main_folder_path + name)]
    for folder in folders:
        # create the file path
        file_pth = main_folder_path + folder + '/'
        # get all the files
        files = [file_pth+name for name in os.listdir(file_pth) if name.endswith(".json")]
        v_sig_index, s_sig_index, windows_val, v_vals, s_vals = overlaying_analysis(files[0],files[1])
        dir = os.path.dirname(files[0])
        ccn_visualization.visualize_still_and_video(dir, '/overlayed_avg_accuracy.png', v_sig_index, s_sig_index, windows_val, v_vals,
                                  s_vals)

if __name__ == "__main__":
    overlay_all("/home/sena/Desktop/Experiments/")

