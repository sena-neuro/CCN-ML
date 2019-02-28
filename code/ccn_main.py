import ccn_io
import ccn_preprocess
import ccn_algorithms
import ccn_ml
import ccn_visualization

har_motion = ["data/human_motion_epochs.mat","data/android_motion_epochs.mat","data/robot_motion_epochs.mat"]
har_still  = [ "data/human_still_epochs.mat", "data/android_still_epochs.mat", "data/robot_still_epochs.mat"]

#------------------ Nope -----------------------
#h = [  "data/human_motion_epochs.mat",  "data/human_still_epochs.mat"]
#a = ["data/android_motion_epochs.mat","data/android_still_epochs.mat"]
#r = [  "data/robot_motion_epochs.mat",  "data/robot_still_epochs.mat"]

combinations = [har_motion, har_still]#, h,a,r]
timeframe = True
#if __name__ == '__main__':
if timeframe:
	print("Checking for different timeframes.\n")
	# check for different timeframes of windows
	accuracies = []
	steps = [125]
	#steps = [100]
	for step_ind in steps:
		x,y = ccn_algorithms.create_windowed_data(["data/android_motion_epochs.mat","data/human_motion_epochs.mat"], 
		              0, timeframe_start = 100, timeframe_end= 800,step=step_ind)
		#x,y = ccn_algorithms.create_singly_windowed(["data/android_motion_epochs.mat","data/human_motion_epochs.mat","data/robot_motion_epochs.mat"], 0, timeframe_start = 100, timeframe_end = 800,step=step_ind)
		x_train, x_test, y_train, y_test = ccn_preprocess.preprocess(x,y, scaler=True, transform=False)
		#accuracy = ccn_ml.logistic_reg(x_train,y_train, x_test,y_test)
		#accuracy = ccn_ml.improved_svm(x_train,y_train, x_test,y_test)
		accuracy = ccn_ml.simple_svm(x_train,y_train, x_test,y_test)
		accuracies.append(accuracy)
		print(step_ind)
		print(accuracy)
	print(accuracies)
	ccn_visualization.plot(steps,accuracies)

if timeframe == False:
	# check for different separate time ranges
	print("Checking for different time ranges.\n")
	accuracies = []
	time_ranges = []
	for time in range(0,700,100):
		
		x,y = ccn_algorithms.create_windowed_data(["data/android_motion_epochs.mat","data/human_motion_epochs.mat"], 
		               0, timeframe_start = time , timeframe_end= time+100,step=50)
		x_train, x_test, y_train, y_test = ccn_preprocess.preprocess(x,y)
		accuracy = ccn_ml.simple_svm(x_train,y_train, x_test,y_test)
		print(time)
		print(accuracy)
		accuracies.append(accuracy)
		time_ranges.append(time)
		
	print(accuracies)
	ccn_visualization.plot(time_ranges,accuracies)
