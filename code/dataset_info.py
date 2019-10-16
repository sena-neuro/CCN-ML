import argparse
import os
import numpy as np
import pandas as pd
import scipy.io as sio

parser = argparse.ArgumentParser(description="Given input path, this script will scan each subjects .mat files"
                                             "and will create an image with trial numbers")

parser.add_argument('input_path', metavar='input path', type=str,
                    help='Input of data path of the experiment')

parser.add_argument('save_path', type=str,
                    help='Where to save the resulting csv file')

parser.add_argument('name', type=str, help="Name of the resulting csv file")
args = parser.parse_args()

dataFileList = []
subjFileList = []
print(args)

# name of folders to take the information from
folders = [name for name in os.listdir(args.input_path) if os.path.isdir(args.input_path + name)]

# get first 6 characters - ex: 'subj02'
subjNameList = []

for folder in folders:

    # create the file path
    file_pth = args.input_path + folder + '/'

    # get all the files
    files = [name for name in os.listdir(file_pth)]
    flag = False

    # get the mat files
    for file in files:
        if file.endswith('.mat'):
            subjFileList.append(file_pth + file)
    if subjFileList:
        dataFileList.append(subjFileList)
        subjNameList.append(folder[0:6])
    subjFileList = []

robot_trial_numbers = []
android_trial_numbers = []
human_trial_numbers = []
subject_total_trial_numbers = []

for subjectFileList in dataFileList:
    subject_total_trial = 0
    for fileName in subjectFileList:
        content = sio.loadmat(fileName)
        if "eeg_data" in content.keys():
            key = "eeg_data"
        elif "prior_still_eeg_data" in content.keys():
            key = "prior_still_eeg_data"
        else:
            key = list(filter(lambda x: "subj" in x, content.keys()))[0]

        if fileName.find('human') >= 0:
            human_trial_numbers.append(content[key].shape[2])
        elif fileName.find('android') >= 0:
            android_trial_numbers.append(content[key].shape[2])
        elif fileName.find('robot') >= 0:
            robot_trial_numbers.append(content[key].shape[2])
        else:
            raise Exception("Couldn't find a label")
        subject_total_trial += content[key].shape[2]
    subject_total_trial_numbers.append(subject_total_trial)

df = pd.DataFrame(list(zip(human_trial_numbers, android_trial_numbers,
                      robot_trial_numbers, subject_total_trial_numbers)),
             index=subjNameList, columns=['No of Human Trials', 'No of Android Trials',
                                      'No of Robot Trials', 'Total Trials'])

df.to_csv(args.save_path + args.name + '.csv')
