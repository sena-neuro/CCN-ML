#
# import needed libraries
#
import ccn_preprocess
import ccn_algorithms
import ccn_ml
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import json


# Generalized Pipeline
# Currently, only SVC is available.
# kernel parameter is only applicable when the method is SVC.
def generalized_pipeline(subjFileList, start=0, end=300, method='svc', \
                         gridsearch=True, window_size=100, window_shift=50,\
                         kernel='rbf', verbose=True, plot=False, minTrials=0):
    """
        Create the information flow for proper calculations.
    """

    # check for different separate time ranges
    accuracies = []
    time_ranges = []

    for time in range(start, end, window_shift):
        if not (time + window_size > end):
            
            try:
                if verbose:
                    print("window [{} - {}]".format(time, time + window_size))
                # use information
                x, y = ccn_algorithms.create_windowed_data(subjFileList, 0,
                                                       timeframe_start=time, timeframe_end=time + window_size,
                                                       size=window_size,
                                                       trials_end="end",minTrials=minTrials)
            except Exception as e:
                print(e)

            x_train, x_test, y_train, y_test = ccn_preprocess.preprocess(x, y,method = "MinMaxScaler")
            #print(subjFileList)
            #print("all data: ", len(x))
            #print("number of training data: ", len(x_train))
            #print("number of test data: ", len(x_test))

            if method   == 'svc':
                accuracy = ccn_ml.svc(x_train, y_train, x_test, y_test, gridsearch=gridsearch, kernel=kernel,
                                      verbose=verbose)
            elif method == "lc":
                accuracy = ccn_ml.logistic_reg(x_train, y_train, x_test, y_test)
            elif method == "dt":
                accuracy = ccn_ml.decision_tree(x_train, y_train, x_test, y_test)

            accuracies.append(accuracy)
            time_ranges.append(time)
    if plot:
        plt.plot(time_ranges, accuracies)
    if verbose:
        print(accuracies)
    return accuracies, time_ranges


def main():
    acc_mat = []
    parser = argparse.ArgumentParser(
        description="For each subject data, divide  to windows with specified size and shift.\n" \
                    "Run classification on each window and find accuracies for each window\n " \
                    "Average accuracies across subjects and plot the average accuracies.")
    parser.add_argument('w_size', metavar='window size', type=int, default=100,
                        help='The window size (default is 100)')
    parser.add_argument('shift', metavar='window shift', type=int, default=0,
                        help='The window size (default is 0)')
    parser.add_argument('minTrials', metavar='minimum trial number', type=int,
                        help='The minimum trial limit for rejection of a subject')
    parser.add_argument('--gridsearch', default=False, action="store_true",
                        help='Perform gridsearch')
    parser.add_argument('-v', '--verbose', default=False, action="store_true",
                        help='Give useful information for debugging')
    parser.add_argument('in_path', metavar='input path',type=str, default='',
                        help='Input of data path of the experiment')
    parser.add_argument('save_path', metavar='output path',type=str, default='',
                        help='Path to save the results of the experiment')
    args = parser.parse_args()
    dataFileList = []
    subjFileList = []

    # name of folders to take the information from
    folders = [name for name in os.listdir(args.in_path) if os.path.isdir(args.in_path + name)]

    # get first 6 characters - ex: 'subj02'    
    subjList = [name[0:6] for name in folders]

    for folder in folders:
        # create the file path
        file_pth = args.in_path + folder + '/'

        # get all the files
        files = [name for name in os.listdir(file_pth)]

        # get the mat files
        for file in files:
            if file.endswith('.mat' ):
                subjFileList.append(file_pth + file)
        dataFileList.append(subjFileList)
        subjFileList = []
    survivingSubjects = len(dataFileList)

    # check if shifting is smaller than size
    assert args.shift <= args.w_size

    # if no shift, then continue
    if args.shift == 0:
        args.shift = args.w_size
    # run code
    for subjectNo, subjFileList in enumerate(dataFileList):
        
        try:
            acc, time_ranges = generalized_pipeline(subjFileList, method = "svc",start=0,end=400, window_size=args.w_size, window_shift=args.shift, gridsearch=args.gridsearch, verbose=args.verbose, minTrials=args.minTrials)
            
        except Exception as e:
            print(e)
            print("Subject {} is rejected due to number of rejected trials ".format(subjList[subjectNo]))
            survivingSubjects -= 1
            continue

        # store results
        subj_acc_dict = {str((range_start, range_start+ args.w_size)): acc[i] for i, range_start in enumerate(time_ranges)}
        results_dict[subjList[subjectNo]+'_results'] = subj_acc_dict
        plt.plot(time_ranges, acc)
        plt.savefig(args.save_path + subjList[subjectNo] + '_accuracy.png',
                    bbox_inches='tight')
        plt.clf()
        plt.close()
        acc_mat.append(acc)

    # store in the end
    print("{} subjects survived trial rejection".format(survivingSubjects))
    acc_mat = np.array(acc_mat)
    avg_accuracies = list(np.mean(acc_mat, axis=0))
    print(time_ranges)
    print(avg_accuracies)
    results_dict['avg_all'] = {str((range_start,range_start+ args.w_size)): avg_accuracies[i] for i, range_start in enumerate(time_ranges)}

    # plot found results
    plt.plot(time_ranges, avg_accuracies)
    plt.savefig(args.save_path + 'avg_accuracy.png', bbox_inches='tight')
    plt.clf()
    plt.close()
    filename = args.save_path + 'accuracy_results.json'
    with open(filename, 'w') as f:
        json.dump(results_dict, f)

# run
if __name__ == '__main__':
    results_dict = {}
    main()
