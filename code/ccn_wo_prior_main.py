import pathlib

import ccn_algorithms
import ccn_ml
import ccn_preprocess
import ccn_stats
import ccn_visualization
import matplotlib as mpl
import numpy as np

mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import json
from sklearn.metrics import confusion_matrix


# Generalized Pipeline
# Currently, only SVC is available.
# kernel parameter is only applicable when the method is SVC.
def generalized_pipeline(subjFileList, start, end, window_size, window_shift, min_trials, gridsearch, method='svc',
                         kernel='rbf', verbose=True):
    # check for different separate time ranges
    accuracies = []
    time_ranges = []
    cms = []
    for time in range(start, end, window_shift):
        if not (time + window_size > end):

            print("window [{} - {}]".format(time, time + window_size))
            # use information
            x, y = ccn_algorithms.create_windowed_data(subjFileList, timeframe_start=time,
                                                       timeframe_end=time + window_size, size=window_size,
                                                       min_trials=min_trials)

            x_train, x_test, y_train, y_test = ccn_preprocess.preprocess(x, y, method="StandardScaler")

            if method == 'svc':
                accuracy, y_pred = ccn_ml.svc(x_train, y_train, x_test, y_test, gridsearch=gridsearch, kernel=kernel,
                                              verbose=verbose)

                # Compute confusion matrix
                cms.append(confusion_matrix(y_test, y_pred).tolist())

            accuracies.append(accuracy)
            time_ranges.append(time)
            if verbose:
                print(accuracy)
    return accuracies, time_ranges, cms


def main():
    results_dict = {}
    acc_mat = []
    cm_mat = []
    classes = ['human', 'android', 'robot']
    parser = argparse.ArgumentParser(
        description="For each subject data, divide  to windows with specified size and shift.\n"
                    "Run classification on each window and find accuracies for each window\n "
                    "Average accuracies across subjects and plot the average accuracies.")

    parser.add_argument('start', metavar='starting milisecond', type=int,
                        help='The ms you want to start the data analysis')
    parser.add_argument('end', metavar='ending milisecond', type=int,
                        help='The ms you want to end the data analysis')
    parser.add_argument('w_size', metavar='window size', type=int,
                        help='The window size (default is 100)')
    parser.add_argument('shift', metavar='window shift', type=int,
                        help='The window size (default is 0)')
    parser.add_argument('min_trials', metavar='minimum trial number', type=int,
                        help='The minimum trial limit for rejection of a subject')
    parser.add_argument('--gridsearch', default=False, action="store_true",
                        help='Perform gridsearch')
    parser.add_argument('-v', '--verbose', default=False, action="store_true",
                        help='Give useful information for debugging')
    parser.add_argument('in_path', metavar='input path', type=str,
                        help='Input of data path of the experiment')
    parser.add_argument('save_path', metavar='output path', type=str,
                        help='Path to save the results of the experiment')
    parser.add_argument('input_type', type=str, choices=['Still', 'Video'], help="The type of the input still or video")

    parser.add_argument('target_labels', type=str, choices=['hra', 'hr', 'ra', 'ah'],
                        help="h for human, a for android and r for robot,"
                             " to distinguish between 2 way classifications and labels"
                             "e.g if hr is entered only human and robot mat files will be"
                             "used for input")

    args = parser.parse_args()
    dataFileList = []
    subjFileList = []
    print(args)
    # name of folders to take the information from
    folders = [name for name in os.listdir(args.in_path) if os.path.isdir(args.in_path + name)]

    # get first 6 characters - ex: 'subj02'
    subjList = [name[0:6] for name in folders]

    for folder in folders:
        # create the file path
        file_pth = args.in_path + folder + '/'

        # get all the files
        files = [name for name in os.listdir(file_pth)]

        flag = False
        chance_level = 0.5
        # get the mat files
        for file in files:
            if args.target_labels == 'hra':
                chance_level = 0.333
                flag = file.endswith('.mat')
            elif args.target_labels == 'hr':
                flag = file.endswith('robot.mat') or file.endswith('human.mat')
            elif args.target_labels == 'ra':
                flag = file.endswith('android.mat') or file.endswith('robot.mat')
            elif args.target_labels == 'ah':
                flag = file.endswith('android.mat') or file.endswith('human.mat')
            if flag:
                subjFileList.append(file_pth + file)
        if subjFileList:
            dataFileList.append(subjFileList)
        subjFileList = []
    survivingSubjects = len(dataFileList)

    # if no shift, then continue
    if args.shift == 0:
        args.shift = args.w_size
    for subjectNo, subjFileList in enumerate(dataFileList):
        if subjFileList:
            try:
                acc, time_ranges, cms = generalized_pipeline(subjFileList, start=args.start, end=args.end,
                                                             window_size=args.w_size,
                                                             window_shift=args.shift, min_trials=args.min_trials,
                                                             gridsearch=args.gridsearch, verbose=args.verbose)

                # store results
                # Confusion matricies per subject
                subj_cm_dict = {str((range_start, range_start + args.w_size)): cms[i] for i, range_start in
                                enumerate(time_ranges)}
                # Accuracies per subject
                subj_acc_dict = {str((range_start, range_start + args.w_size)): acc[i] for i, range_start in
                                 enumerate(time_ranges)}

                # Add cms and accuracies to a bigger dictionary
                results_dict[subjList[subjectNo] + 'accuracy_results'] = subj_acc_dict
                results_dict[subjList[subjectNo] + 'cms'] = subj_cm_dict

                # Create target Directory if don't exist
                if not os.path.exists(args.save_path + args.target_labels + "_" + "confusion_matrices/"):
                    os.mkdir(args.save_path + args.target_labels + "_" + "confusion_matrices/")

                # Plot and save the accuracy graph per subject
                plt.plot(time_ranges, acc)
                plt.savefig(args.save_path + args.input_type + "_" + subjList[subjectNo] + "_" +
                            args.target_labels + '_accuracy.png',
                            bbox_inches='tight')
                plt.clf()
                plt.close()

                # Create bigger matricies to hold accuracy and cm information for averaging ease
                acc_mat.append(acc)
                cm_mat.append(cms)
            except AssertionError as err:
                print(err)
                survivingSubjects -= 1
                continue
            except ValueError as valerr:
                print(valerr)
                exit()

    # store in the end
    print("{} subjects survived trial rejection".format(survivingSubjects))
    if survivingSubjects != 0:
        total_cms = np.sum(np.array(cm_mat), axis=0)
        avg_cms = np.mean(np.array(cm_mat), axis=0)
        avg_accuracies = np.mean(np.array(acc_mat), axis=0).tolist()

        results_dict['avg_accuracies'] = {
            str((2 * (range_start - 100), 2 * (range_start + args.w_size - 100))): avg_accuracies[i] for
            i, range_start in enumerate(time_ranges)}
        results_dict['total_cm'] = {
            str((2 * (range_start - 100), 2 * (range_start + args.w_size - 100))): total_cms[i].tolist() for
            i, range_start in enumerate(time_ranges)}
        results_dict['avg_cm'] = {
            str((2 * (range_start - 100), 2 * (range_start + args.w_size - 100))): avg_cms[i].tolist() for
            i, range_start in enumerate(time_ranges)}
        results_dict['target_labels'] = args.target_labels

        # plot found results
        plt.plot(time_ranges, avg_accuracies)
        plt.savefig(args.save_path + args.input_type + "_" + args.target_labels + '_avg_accuracy.png',
                    bbox_inches='tight')
        plt.clf()
        plt.close()
        for time_range, cm in results_dict['avg_cm'].items():
            ccn_visualization.plot_confusion_matrix(
                cm, classes, normalize=True,
                title=" Average Confusion Matrix On Time Window: " + time_range)
            plt.savefig(args.save_path + args.target_labels + "_" + "confusion_matrices/" + args.input_type + "_" +
                        time_range + "_" + args.target_labels + '_avg_cm.png',
                        bbox_inches='tight')
            plt.clf()
            plt.close()
        for time_range, cm in results_dict['total_cm'].items():
            ccn_visualization.plot_confusion_matrix(
                cm, classes, normalize=True,
                title=" Average Confusion Matrix On Time Window: " + time_range)
            plt.savefig(args.save_path + args.target_labels + "_" + "confusion_matrices/" + args.input_type + "_" +
                        time_range + "_" + args.target_labels + '_total_cm.png',
                        bbox_inches='tight')
            plt.clf()
            plt.close()
        filename = args.save_path + args.input_type + "_" + args.target_labels + '_accuracy_results.json'
        with open(filename, 'w') as f:
            json.dump(results_dict, f)

    # Get the main path of experiments from save path
    exp_path = pathlib.PurePath(args.save_path)

    # Give this path to the statistic module as main path
    parent_path = str(exp_path.parent)
    print(type(parent_path))
    ccn_stats.overlay_all(parent_path + "/", chance_level)


if __name__ == '__main__':
    main()
