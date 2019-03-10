import ccn_io
import ccn_preprocess
import ccn_algorithms
import ccn_ml
import ccn_visualization

# check for different separate time ranges
accuracies = []
time_ranges = []
shift = 50
size = 200
start = 0
end = 600
assert shift <= size
dataFileList = ["data2/video_android.mat","data2/video_human.mat","data2/video_robot.mat"]

for time in range(start,end,shift):
	if not (time + size > end):
		print("window [{} - {}]".format(time, time+size))
		x,y = ccn_algorithms.create_windowed_data(dataFileList, 0,
			timeframe_start=time, timeframe_end=time+size,size=size, trials_end=90)
		print("all data: ", len(x))
		x_train, x_test, y_train, y_test = ccn_preprocess.preprocess(x,y)
		print("number of training data: ", len(x_train))
		print("number of test data: ", len(x_test))
		accuracy = ccn_ml.improved_svm(x_train,y_train, x_test,y_test)
		accuracies.append(accuracy)
		time_ranges.append(time)

print(accuracies)
#ccn_visualization.plot(time_ranges,accuracies)
