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
from scipy.stats import ttest_1samp, ttest_ind
from statsmodels.sandbox.stats.multicomp import multipletests


def read_and_prepare_data(file):
    # read data
    with open(file) as f:
        data = json.load(f)

    avg_accuracies = [[k, v] for k, v in data['avg_accuracies'].items()]
    del (data['avg_accuracies'])

    target_labels = data['target_labels']

    # keys for json object
    subjects = [key for key in data.keys() if key.endswith('accuracy_results')]
    windows = list(data[subjects[0]].keys())

    # information storage
    eeg_sliced = {}
    for each_window in windows:
        eeg_sliced[each_window] = []  # initialization with empty list for each window of eeg data

    # store info of of all subjects per window
    for window in windows:
        for subject in subjects:
            eeg_sliced[window].append(data[subject][window])

    return avg_accuracies, eeg_sliced, target_labels


def compare_with_chance_level(vector1, chance_level):
    windows = list(vector1.keys())

    ttest_values = []
    for window in windows:
        ttest_values.append(ttest_1samp(np.asarray(vector1[window]),chance_level))

    # combine info
    t_values = np.asarray(ttest_values)
    p_values = [float(i) for i in t_values[:, -1]]

    # correction
    p_adjusted = multipletests(p_values, method='bonferroni')[:2]
    p_adjusted_l = list(map(list, zip(*[x.tolist() for x in p_adjusted])))

    # significant indices
    sig_indices = p_adjusted[0]

    return sig_indices


def compare_vectors(vector1, vector2):
    '''Find significantly different time windows'''
    # TODO vector is misleading, change it to dict or something

    windows = list(vector1.keys())

    ttest_values = []
    for window in windows:
        ttest_values.append(ttest_ind(np.asarray(vector1[window]), np.asarray(vector2[window])))

        # combine info
    t_values = np.asarray(ttest_values)
    p_values = [float(i) for i in t_values[:, -1]]

    # correction
    p_adjusted = multipletests(p_values, method='bonferroni')[:2]
    p_adjusted_l = list(map(list, zip(*[x.tolist() for x in p_adjusted])))

    # significant indices
    sig_indices = p_adjusted[0]

    return sig_indices


def choose_from_options(list_options, replace_option=""):
    """Printing options the easily for the user to use."""

    separator = "\n-------------------------------------------------------\n"

    for index in range(len(list_options)):
        print("[" + str(index + 1) + "] " + list_options[index].replace(replace_option, ""))

    choice = int(input('Enter choice: '))
    print("\n")

    return choice - 1


def choose(folder_list, chance_level):
    """Choose the accuracies you want to work with."""

    # get directory you want to work with
    folder_num = choose_from_options(folder_list)

    # get the file you want to work with
    folders = glob.glob(folder_list[folder_num] + "*/accuracy_results.json")
    subdir = choose_from_options(folders, folder_list[folder_num])

    # choose
    print("You chose this file for analysis: " + folders[subdir] + ' .')
    response = input('Do you want to continue?(Y/anything else)')

    if response not in ['Y', 'y']:
        return

    else:
        avg_values, eeg_sliced = read_and_prepare_data(folders[subdir])
        sig_index = compare_with_chance_level(eeg_sliced, chance_level)

        windows = list(eeg_sliced.keys())
        dir = os.path.dirname(folders[subdir])  ## directory of file
        ccn_visualization.visualize(dir, '/avg_accuracy.png', sig_index, windows, avg_values, chance_level)


def overlay(video_path, still_path, chance_level):
    "Takes paths to results json of video and still input experiments, and overlays the significance analysis"

    v_avg_vals, v_eeg_sliced, v_target_labels = read_and_prepare_data(video_path)
    v_sig_index = compare_with_chance_level(v_eeg_sliced, chance_level)

    s_avg_vals, s_eeg_sliced, s_target_labels = read_and_prepare_data(still_path)
    s_sig_index = compare_with_chance_level(s_eeg_sliced, chance_level)

    v_eeg_windows = list(v_eeg_sliced.keys())

    if v_target_labels != s_target_labels:
        raise Exception('Still and window targets are not equal e.g human android vs human robot')

    # Check if video and still files have the same windows, otherwise t-test would be wrong
    if v_eeg_windows != list(s_eeg_sliced.keys()):
        raise Exception('Still and video window values are not equal')

    sig_list = compare_vectors(v_eeg_sliced, s_eeg_sliced)
    sig_windows = [v_eeg_windows[index] for index, sig in enumerate(sig_list) if sig]
    if sig_windows:
        experiment_name = video_path.split("/")[-2]
        print("Still and video data is different at windows: ")
        print(sig_windows)
        print("at experiment: "+ experiment_name)

    save_dir = os.path.dirname(video_path)  ## directory of file
    ccn_visualization.visualize_still_and_video(save_dir, v_target_labels + '_overlayed_avg_accuracy.png', v_sig_index, s_sig_index, v_eeg_windows,
                                                v_avg_vals, s_avg_vals, chance_level)


def run_all(folder_list, chance_level, overlaying=False):
    # Run all the information that is in each folder.
    for each in folder_list:
        folders = glob.glob(each + "*/accuracy_results.json")
        for each_subdir in folders:
            # print(each_subdir)
            avg_values, eeg_sliced = read_and_prepare_data(each_subdir)
            sig_index = compare_with_chance_level(eeg_sliced, chance_level)

            windows = list(eeg_sliced.keys())
            dir = os.path.dirname(each_subdir)  ## directory of file
            ccn_visualization.visualize(dir, '/avg_accuracy.png', sig_index, windows, avg_values, chance_level)


def overlay_all(main_folder_path, chance_level):  # The main folder's path which includes all experiments
    """

    :type main_folder_path: string
    """
    # name of folders to take the information

    print("path in overlay all: ", main_folder_path)
    folders = [name for name in os.listdir(main_folder_path) if os.path.isdir(main_folder_path + name)]
    for folder in folders:

        # create the file path
        file_pth = main_folder_path + folder + '/'

        # get all the files
        files = [file_pth + name for name in os.listdir(file_pth) if name.endswith(".json")]
        if len(files) == 2:
            if "Still" in files[0]:
                still_file_path = files[0]
                video_file_path = files[1]
            else:
                still_file_path = files[1]
                video_file_path = files[0]
            overlay(video_file_path, still_file_path, chance_level)
        else:
            print("Error: There needs to be at least two json files")

if __name__ == '__main__':
    overlay_all('/Users/huseyinelmas/Desktop/Experiments/desktop_experiments/')
