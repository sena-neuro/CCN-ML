import ccn_io
import ccn_preprocess
import ccn_algorithms
import ccn_ml
import ccn_visualization

# check for different separate time ranges
accuracies = []
time_ranges = []
window_size = 500
for time in range(100, 600, window_size):
    print(time)
    x, y = ccn_algorithms.create_windowed_data(
        ["data2/video_android.mat", "data2/video_human.mat", "data2/video_robot.mat"],
        0, timeframe_start=time, timeframe_end=time + window_size, step=100, trials_end=90)  # step was 50

    x_train, x_test, y_train, y_test = ccn_preprocess.preprocess(x, y)
    accuracy = ccn_ml.sena_svc(x_train, y_train, x_test, y_test)
    print(time)
    print(accuracy)
    accuracies.append(accuracy)
    time_ranges.append(time)
print(accuracies)
ccn_visualization.plot(time_ranges, accuracies)

