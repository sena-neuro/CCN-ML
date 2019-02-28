#!/usr/bin/env python
# coding: utf-8

# Author: Doren Calliku
# Lab:    CCN

#----------------------------------------------------------------
# useful libraries for math and reading files
import numpy as np
#import pandas as pd
import scipy.io as sio
#import itertools

# preprocessing
#from sklearn import metrics
#from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# plotting
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches

# svm is implemented using libsvm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# specialty for EEG Data
#import mne
#-----------------------------------------------------------------
# Useful Methods
def read_file(fname = "data.mat"):
    """
        Reading mat files. Use the scipy library or other library depending on the file type.  
    """
    #list_properties = sio.whosmat(fname)
    #content = sio.loadmat(fname)['a']
    content = sio.loadmat(fname)
    return content

def preprocess(x,y,train_test=True, scaler = True, transform= False):
    """
    Preprocessing steps: 
    1. MinMax Scaler
    2. Transform 
    3. Train test split
    """
    # WORK HERE@
    if (scaler):
        scaler = MinMaxScaler() # Standard scaler too 
        scaler.fit(x)
        x = scaler.transform(x)
    
    # simple transformation which takes out the average
    if transform:
        data_mean = x.mean()
        data_std = x.std()
        x = (x - data_mean)/data_std
    
    if train_test:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
        return x_train, x_test, y_train, y_test
    
    else:
        return x,y

def create_windowed_data(filenames,
    test_type=0,
    channels_start=0,
    channels_end=64,
    timeframe_start=100,
    timeframe_end=1100,
    trials_start=0,
    trials_end=50,
    step=100
    ):

    # output for usage
    x = []
    y = []

    # for each of the files
    for filename in filenames:
	
		# read the contents
        content    = read_file(filename)
        class_type = list(content.keys())[-1] + ''
        data_x     = np.asarray(content[class_type])[ channels_start : channels_end, timeframe_start : timeframe_end,  trials_start : trials_end]
        vector     = []
        window_frame = timeframe_start

        # get all the information for each of the trials
        for trial in range(trials_start, trials_end):
            # for each of the window frames that you have decided to work on
            while window_frame + step <= timeframe_end:
                # select the part of data you want 
                vector = np.array( data_x[channels_start:channels_end, window_frame: window_frame + step, trial])
                
                # 2d array to 1D with attachment one after the other. 
                vector = vector.flatten()
                
                # check if step is opening any problems
                if vector.size >= (step * (channels_end - channels_start)):
                    
                    # add data for ml
                    x.append(vector.tolist())
                    
                    # categorize for statistical process
                    if class_type.find('human') >= 0:
                        y.append(0)
                    elif class_type.find('android') >= 0:
                        y.append(1)
                    else:
                        y.append(2)
                # loop the while statement
                window_frame += step
                
            # keep the while loop running for next run of for loop
            window_frame = 0
    return x,y


# SVM
def simple_svm(x_train,y_train, x_test, y_test):
    svm_model_linear = SVC(kernel = 'linear',verbose=1).fit(x_train, y_train) 
    #svm_predictions = svm_model_linear.predict(x_test) 
    # model accuracy for X_test   
    accuracy = svm_model_linear.score(x_test, y_test) 
    #print("Accuracy of the model:",accuracy)
    # creating a confusion matrix 
    # cm = confusion_matrix(y_test, svm_predictions)
    #print("The confusion matrix:\n",cm)
    return accuracy

def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.predict()
    grid_search.best_params_
    return grid_search.best_params_

def improved_svm(X_train, y_train, X_test, y_test, multiclass = False):
    
    param_grid = [
      {'C': [0.1,0.7,0.8,0.9,1], 'kernel': ['linear']},
      {'C': [0.9,1, 1.1], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
     ]
    grid_search = GridSearchCV(SVC(), param_grid, cv=2)
    grid_search.fit(X_train, y_train)
    #predicted   = grid_search.predict(X_test)
    print("Best parameters: ",grid_search.best_params_)
    #print("Accuracy = {}\n".format( metrics.accuracy_score(y_test, predicted)))
    # creating a confusion matrix 
    # cm = confusion_matrix(y_test, predicted)
    #print(cm)


# Apply model

har_motion = ["data/human_motion_epochs.mat","data/android_motion_epochs.mat","data/robot_motion_epochs.mat"]
har_still  = [ "data/human_still_epochs.mat", "data/android_still_epochs.mat", "data/robot_still_epochs.mat"]
#------------------ Nope -----------------------
h = [  "data/human_motion_epochs.mat",  "data/human_still_epochs.mat"]
a = ["data/android_motion_epochs.mat","data/android_still_epochs.mat"]
r = [  "data/robot_motion_epochs.mat",  "data/robot_still_epochs.mat"]
combinations = [har_motion, har_still]#, h,a,r]

accuracies = []
steps = [100]

for step_ind in steps:
	print("1.")
	x,y = create_windowed_data(["data/android_motion_epochs.mat","data/human_motion_epochs.mat"], 
		           0, timeframe_start = 100, timeframe_end= 600,step=step_ind)
	print("2.")
	x_train, x_test, y_train, y_test = preprocess(x,y)
	print("3.")
	accuracy = simple_svm(x_train,y_train, x_test,y_test)
	print("4.")
	accuracies.append(accuracy)

print(accuracies)

"""
accuracies = []
start = []

for time in range(0,700,100):
    
    x,y = create_windowed_data(["data/android_motion_epochs.mat","data/human_motion_epochs.mat"], 
                   0, timeframe_start = time , timeframe_end= time+100,step=50)
    
    x_train, x_test, y_train, y_test = preprocess(x,y)
    accuracy = simple_svm(x_train,y_train, x_test,y_test)
    accuracies.append(accuracy)
    start.append(time)
    
print(accuracies)


plt.plot(start,accuracies)

"""
# In[ ]:




