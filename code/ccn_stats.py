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
from scipy.stats import sem
from scipy.stats import ttest_1samp, ttest_ind
from statsmodels.sandbox.stats.multicomp import multipletests


def read_and_prepare_data(file):
    # read data
    with open(file) as f:
        data = json.load(f)

    avg_accuracies = [[k, v] for k, v in data['avg_accuracies'].items()]
    del (data['avg_accuracies'])

    exp_params = data['experiment_params']


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

    return avg_accuracies, eeg_sliced, exp_params


def compare_with_chance_level(vector1, chance_level):
    windows = list(vector1.keys())

    p_values = []
    t_values = []
    for window in windows:
        t_statistic, p_value = ttest_1samp(vector1[window], chance_level)
        p_values.append(p_value/2)
        t_values.append(t_statistic)

    # correction
    p_adjusted = multipletests(p_values, method='bonferroni', alpha=0.05)

    # one-tailed
    for i,t in enumerate(t_values):
        if t < 0:
            p_adjusted[0][i] = False


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


def choose(folder_list):
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
        avg_values, eeg_sliced, target_labels = read_and_prepare_data(folders[subdir])
        if target_labels == 'hra':
            chance_level = 0.333
        else:
            chance_level = 0.5
        sig_index = compare_with_chance_level(eeg_sliced, chance_level)

        windows = list(eeg_sliced.keys())
        dir = os.path.dirname(folders[subdir])  ## directory of file
        ccn_visualization.visualize(dir, '/avg_accuracy.png', sig_index, windows, avg_values, chance_level)


def overlay(first_path, second_path):
    "Takes paths to results json of video and still input experiments, and overlays the significance analysis"

    first_avg_vals, first_eeg_sliced, first_exp_params = read_and_prepare_data(first_path)
    second_avg_vals, second_eeg_sliced, second_exp_params = read_and_prepare_data(second_path)
    first_eeg_windows = list(first_eeg_sliced.keys())

    # Calculate standard error of the mean for each window
    first_sems = [sem(first_subj_acc_list) for window, first_subj_acc_list in first_eeg_sliced.items()]
    second_sems = [sem(second_subj_acc_list) for window, second_subj_acc_list in second_eeg_sliced.items()]

    # TODO make a more complete checking
    if first_exp_params['target_labels'] != second_exp_params['target_labels']:
        raise Exception('Given targets are not the same e.g human android vs human robot')

    if first_exp_params['target_labels'] == 'hra':
        chance_level = 0.333
    else:
        chance_level = 0.5

    first_sig_index = compare_with_chance_level(first_eeg_sliced, chance_level)
    second_sig_index = compare_with_chance_level(second_eeg_sliced, chance_level)

    # Check if video and still files have the same windows, otherwise t-test would be wrong
    if first_eeg_windows != list(second_eeg_sliced.keys()):
        raise Exception('Still and video window values are not equal')

    sig_list = compare_vectors(first_eeg_sliced, second_eeg_sliced)
    sig_windows = [first_eeg_windows[index] for index, sig in enumerate(sig_list) if sig]
    if sig_windows:
        experiment_name = first_path.split("/")[-2]
        print(first_exp_params['exp_type'] + "-" + first_exp_params['input_type'] + " and " +
            second_exp_params['exp_type'] + "-" + second_exp_params['input_type'] )
        print(" are differing significantly at time windows: ")
        print(sig_windows)
        print("at experiment: "+ experiment_name)

    #todo ALSO CORRECT OTHERS WINDOWS->WINDOWS_VAL
    windows_val = [2 * (x - 100) for x in [int(wind_frame.strip('()').split(',')[0])
                                           for wind_frame in first_eeg_windows]]


    save_dir = os.path.dirname(first_path)  ## directory of file
    ccn_visualization.visualize_two_curves(save_dir+'/overlayed_avg_accuracy.png', first_sig_index, second_sig_index,
                                                windows_val, first_avg_vals, second_avg_vals, chance_level,
                                                first_sems, second_sems, first_exp_params, second_exp_params)


def run_all(folder_list, chance_level, overlaying=False):
    # Run all the information that is in each folder.

    for each in folder_list:
        folders = glob.glob(each + "*/accuracy_results.json")
        for each_subdir in folders:
            # print(each_subdir)
            avg_values, eeg_sliced, target_labels = read_and_prepare_data(each_subdir)
            if target_labels == 'hra':
                chance_level = 0.333
            else:
                chance_level = 0.5
            sig_index = compare_with_chance_level(eeg_sliced, chance_level)

            windows = list(eeg_sliced.keys())
            dir = os.path.dirname(each_subdir)  ## directory of file
            ccn_visualization.visualize(dir, '/avg_accuracy.png', sig_index, windows, avg_values, chance_level)


def overlay_all(main_folder_path):  # The main folder's path which includes all experiments
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
            overlay(video_file_path, still_file_path)
        else:
            print("Error: There needs to be at least two json files")

if __name__ == '__main__':
    overlay('/home/sena/Desktop/tst/_Naive_Video_hra_accuracy_results.json', '/home/sena/Desktop/tst/_Naive_Still_hra_accuracy_results.json')
