import ccn_io
import ccn_preprocess
import ccn_algorithms
import ccn_ml
import ccn_visualization

# check for different separate time ranges
accuracies = []
time_ranges = []
for time in range(0,600,100):
	print(time)
	x,y = ccn_algorithms.create_windowed_data(["data2/video_android.mat","data2/video_human.mat","data2/video_robot.mat"], 
	               0, timeframe_start = time , timeframe_end= time+100,step=50)

	x_train, x_test, y_train, y_test = ccn_preprocess.preprocess(x,y)
	accuracy = ccn_ml.simple_svm(x_train,y_train, x_test,y_test)
	print(time)
	print(accuracy)
	accuracies.append(accuracy)
	time_ranges.append(time)
print(accuracies)
ccn_visualization.plot(time_ranges,accuracies)