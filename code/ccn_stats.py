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

import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.sandbox.stats.multicomp import multipletests


def analyze(file, create_figures=True):
    """Extract significant information from the json file."""

    # <================================= Data ===============================>
    # read data
    with open(file) as f:
        data = json.load(f)

        # keys for json object
    subjects = list(data.keys())
    windows = list(data[subjects[0]].keys())

    # information storage
    info_storage = {}
    for each in windows:
        info_storage[each] = []  # initialization with empty list

    # store info
    for window in windows:
        for subject in subjects:
            info_storage[window].append(data[subject][window])

    # <================================= Tests ================================>
    # do t-test
    ttest_values = []
    for window in windows:
        ttest_values.append(ttest_1samp(info_storage[window][:-1], 0.333)) # Why [:-1] and not all of info storage

    # combine info
    avg_values = [[k, v] for k, v in data['avg_all'].items()]
    t_values = np.asarray(ttest_values)
    p_values = [float(i) for i in t_values[:, -1]]

    # correction
    p_adjusted = multipletests(p_values, method='bonferroni')[:2]
    p_adjusted_l = list(map(list, zip(*[x.tolist() for x in p_adjusted])))

    # for ease of use - pandas DataFrame
    values = np.concatenate((avg_values, t_values, p_adjusted_l), axis=1)
    data = pd.DataFrame(values)  # pandas
    data.columns = ['window', 'accuracy', 't-statistic', 'pvalue', 'truthfulness', 'new_p']

    # <=========================== visualize ==================================>
    plt.clf()
    fig, ax = plt.subplots()

    # decide on time
    windows_val = [2 * (x - 100) for x in [int(wind_frame.strip('()').split(',')[0])
                                           for wind_frame in windows]]

    # average values
    vals = [x[-1] for x in avg_values][:len(windows_val)]

    ax.plot(windows_val, vals)

    # p significance
    sig_index = data[data['truthfulness'] == '1.0']['accuracy'].index.tolist()[:len(windows_val)]

    if create_figures:
        # print(windows_val)
        ax.plot([windows_val[x] for x in sig_index], [vals[x] for x in sig_index],
                linestyle='none', color='r', marker='o')

        # show starting understanding and chance level
        ax.axvline(x=0, color='black', alpha=0.5, linestyle='--', label='end of baseline period')
        ax.axhline(y=0.33, color='red', alpha=0.5, label='chance level')

        ax.legend(loc='upper right')
        ax.set_title('Classification accuracies')

        dir = os.path.dirname(file)  ## directory of file
        fig.savefig(dir+'/avg_accuracy.png', bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
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
        analyze(folders[subdir])

def overlaying_analysis(video_path, still_path):
    "Takes paths to results json of video and still input experiments, and overlays the significance analysis"

    v_sig_index, v_windows_val, v_vals = analyze(video_path, False)
    s_sig_index, s_windows_val, s_vals = analyze(still_path, False)

    fig, ax = plt.subplots()
    ax.plot(v_windows_val, v_vals,'g', label="Video")
    ax.plot(s_windows_val, s_vals,'b', label='Still')

    # print(windows_val)
    ax.plot([v_windows_val[x] for x in v_sig_index], [v_vals[x] for x in v_sig_index],
            linestyle="none", color='r', marker='o')
    ax.plot([s_windows_val[x] for x in s_sig_index], [s_vals[x] for x in s_sig_index],
            linestyle="none", color='r', marker='o')

    # show starting understanding and chance level
    ax.axvline(x=0, color='black', alpha=0.5, linestyle='--', label='end of baseline period')
    ax.axhline(y=0.33, color='red', alpha=0.5, label='chance level')

    ax.legend(loc='upper right')
    ax.set_title('Classification accuracies')  # If we really want to, we can get
    # the window size and shift from the data
    dir = os.path.dirname(video_path)  ## directory of file
    fig.savefig(dir + '/overlayed_avg_accuracy.png', bbox_inches='tight')
    plt.clf()
    plt.close()

# TODO: Ask what is the folder list and when it is necessary

def run_all(folder_list, overlaying=False):

    # Run all the information that is in each folder.
    for each in folder_list:
        folders = glob.glob(each + "*/accuracy_results.json")
        for each_subdir in folders:
            print(each_subdir)
            #analyze(each_subdir)

def overlay_all(main_folder_path):
    # name of folders to take the information from
    folders = [name for name in os.listdir(main_folder_path) if os.path.isdir(main_folder_path + name)]

    for folder in folders:
        # create the file path
        file_pth = main_folder_path + folder + '/'

        # get all the files
        files = [file_pth+name for name in os.listdir(file_pth) if name.endswith(".json")]
        overlaying_analysis(files[0],files[1])


#folder_list = [ "/Users/huseyinelmas/Desktop/Experiments/Video/"]
#choose(folder_list)
#run_all(folder_list)
#analyze("/Users/huseyinelmas/Desktop/Experiments/old/Video_accuracy_results.json")
#overlaying_analysis("/Users/huseyinelmas/Desktop/Experiments/old/Video_accuracy_results.json",
                   # "/Users/huseyinelmas/Desktop/Experiments/old/Still_accuracy_results.json")

overlay_all("/Users/huseyinelmas/Desktop/Experiments/")

