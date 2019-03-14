import ccn_preprocess
import ccn_algorithms
import ccn_ml
import ccn_visualization

dataFileList = ["data2/video_android.mat", "data2/video_human.mat", "data2/video_robot.mat"]


# Generalized Pipeline
# Currently, only SVC is available.
# kernel parameter is only applicable when the method is SVC.
def generalized_pipeline(start, end, method='svc', gridsearch=True, window_size=100, window_shift=50, kernel='rbf',
                         verbose=True, plot=False):
    # check for different separate time ranges
    accuracies = []
    time_ranges = []
    assert window_shift <= window_size
    for time in range(start, end, window_shift):
        if not (time + window_size > end):
            print("window [{} - {}]".format(time, time + window_size))
            x, y = ccn_algorithms.create_windowed_data(dataFileList, 0,
                                                       timeframe_start=time, timeframe_end=time + window_size,
                                                       size=window_size,
                                                       trials_end=90)
            print("all data: ", len(x))
            x_train, x_test, y_train, y_test = ccn_preprocess.preprocess(x, y)
            print("number of training data: ", len(x_train))
            print("number of test data: ", len(x_test))
            if method == 'svc':
                accuracy = ccn_ml.svc(x_train, y_train, x_test, y_test, gridsearch=gridsearch, kernel=kernel,
                                      verbose=verbose)
            accuracies.append(accuracy)
            time_ranges.append(time)
            if plot:
                ccn_visualization.plot(time_ranges, accuracies)

    print(accuracies)


'''
# First: 100 ms discrete window
print('(1) Data is sliced into 100 ms discrete windows')
generalized_pipeline(start=100, end=600, window_size=100, window_shift=100)
'''

# Second: 100 ms & 50 ms shift window
print('(2) Data is sliced into 100 ms windows, shifted by 50 ms (0-100, 50-150, ...)')
generalized_pipeline(start=100, end=600, window_size=100, window_shift=50)

'''
# Third: 50 ms discrete window
print('(3) Data is sliced into 50 ms discrete windows')
generalized_pipeline(start=100, end=600, window_size=50, window_shift=50)

# Fourth: Full movie window
print('(4) The whole movie duration is taken (No slicing)')
generalized_pipeline(start=100, end=600, window_size=500, window_shift=500)


# Fifth: 100 ms & 20 ms shift window
print('(2) Data is sliced into 100 ms windows, shifted by 20 ms (0-100, 20-120, ...)')
generalized_pipeline(start=100, end=600, window_size=100, window_shift=20)
'''
