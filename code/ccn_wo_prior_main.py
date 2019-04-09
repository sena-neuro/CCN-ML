import ccn_preprocess
import ccn_algorithms
import ccn_ml
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


# Generalized Pipeline
# Currently, only SVC is available.
# kernel parameter is only applicable when the method is SVC.
def generalized_pipeline(subjFileList, start=0, end=400, method='svc', gridsearch=True, window_size=100, window_shift=50, kernel='rbf',
                         verbose=True, plot=False):
    # check for different separate time ranges
    accuracies = []
    time_ranges = []
    assert window_shift <= window_size
    if window_shift == 0:
      window_shift=window_size
    for time in range(start, end, window_shift):
        if not (time + window_size > end):
            print("window [{} - {}]".format(time, time + window_size))
            x, y = ccn_algorithms.create_windowed_data(subjFileList, 0,
                                                       timeframe_start=time, timeframe_end=time + window_size,
                                                       size=window_size,
                                                       trials_end=90)
            x_train, x_test, y_train, y_test = ccn_preprocess.preprocess(x, y)
            if verbose:
              print("all data: ", len(x))
              print("number of training data: ", len(x_train))
              print("number of test data: ", len(x_test))

            if method == 'svc':
              accuracy = ccn_ml.svc(x_train, y_train, x_test, y_test, gridsearch=gridsearch, kernel=kernel,
                                      verbose=verbose)
            accuracies.append(accuracy)
            time_ranges.append(time)
    if plot:
      ccn_visualization.plot(time_ranges, accuracies) 
    if verbose:
      print(accuracies)
    return accuracies, time_ranges

def main():
  acc_mat = []
  parser = argparse.ArgumentParser(description= "For each subject data, divide  to windows with specified size and shift.\n"\
                                                "Run classification on each window and find accuracies for each window\n "\
                                                "Average accuracies across subjects and plot the average accuracies.")
  parser.add_argument('w_size', metavar='window size', type=int ,default=100,
                      help='The window size (default is 100)')
  parser.add_argument('shift', metavar='window shift', type=int,default=0,
                      help='The window size (default is 0)')
  parser.add_argument('--gridsearch', default=False,action="store_true",
                      help='Perform gridsearch')
  parser.add_argument('-v','--verbose',default=False,action="store_true",
                      help='Give useful information for debugging')
  parser.add_argument('save_path', metavar='output path', type=str, default='', 
  					          help='Path to save the results of the experiment' )
  parser.add_argument('user', metavar='user', type=str, default='', 
                      help='In whos server oelmas or ser' )
  svpth = '/auto/data2/' + user +'/EEG_AgentPerception_NAIVE/Analysis/'
  dataFileList = []
  subjFileList = []
  pth = '/auto/data2/' + user +'/EEG_AgentPerception_NAIVE/Data/'
  print(pth)
  folders =  [name for name in os.listdir(pth) if os.path.isdir(pth+name)]
  for folder in folders:
    file_pth = pth + folder+'/'
    files = [name for name in os.listdir(file_pth)]
    for file in files:
      if('.mat' in file):
        print("file path: ", file_pth + file)
        subjFileList.append(file_pth + file)
    dataFileList.append(subjFileList)
    subjFileList = []

  print(dataFileList)
  #dataFileList = [["data2/s02_video_android", "data2/s02_video_human", "data2/s02_video_robot"]]

  args = parser.parse_args()
  
  print('(2) Data is sliced into 100 ms windows, shifted by 50 ms (0-100, 50-150, ...)')
  for subjectNo, subjFileList in enumerate(dataFileList):
    acc,time_ranges=generalized_pipeline(subjFileList, start=0, end=400, window_size=args.w_size, window_shift=args.shift,
                                        gridsearch=args.gridsearch, verbose=args.verbose)
    plt.plot(time_ranges,acc)
    plt.savefig(svpt+args.save_path+'/'+str(args)+'s'+str(subjectNo)+'_accuracy.png',bbox_inches='tight')
    acc_mat.append(acc) 

  acc_mat = np.array(acc_mat)
  avg_accuracies = np.mean(acc_mat, axis=0)
  plt.plot(time_ranges,avg_accuracies)
  plt.savefig(svpth+args.save_path+'/'+str(args)+'_accuracy.png',bbox_inches='tight')
if __name__ == '__main__':
  main()
