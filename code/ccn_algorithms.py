#!/usr/bin/env python3

import numpy as np
import scipy.io as sio


# useful! Contains 3 for loops inside: files, trials, window_frames
def create_windowed_data(filenames,
    test_type=0, channels_start=0, channels_end=62, timeframe_start=100,
    timeframe_end=1100, trials_start=0, trials_end=50, size=100):
    """
        This method creates windowing for each of the channels put to it.
        It is the algorithm which allows comparison between the time trials. 
        No channel windowing present for the moment. To use shif
    """
    # output for usage
    x = []
    y = []

    # for each of the files
    for filename in filenames:

        # read the contents
        content    = sio.loadmat(filename)
        class_type = list(content.keys())[-1] + ''
        data_x     = np.asarray(content[class_type])
        vector     = []
        wind_start = timeframe_start
        label = None
        if filename.find('human') >= 0:
            label = 0
        elif filename.find('android') >= 0:
            label = 1
        elif filename.find('robot') >= 0:
            label = 2
        else:
             raise Exception("Couldn't find a label")
   
        if len(data_x.shape) <3:
            return
        
        if trials_end == "end":
            trials_end = data_x.shape[2]
            
        if channels_end > data_x.shape[0]:
            raise Exception("More channels are requested than in the data")
            
        elif trials_end > data_x.shape[2]:
            trials_end=len(data_x[0][0])

        # get all the information for each of the trials
        for trial in range(trials_start, trials_end):
            # for each of the window frames that you have decided to work on
            while wind_start + size <= timeframe_end:

                # select the part of data you want, 2d array to 1D with attachment one after the other. 

                vector = data_x[channels_start:channels_end, wind_start: wind_start + size, trial] # TODO: only allows a range allow for specific values too

                vector = vector.flatten()

                # check if size is opening any problems
                assert vector.size == (size * (channels_end - channels_start))

                # add data for ml
                x.append(vector)
                y.append(label)

                # loop the while statement
                wind_start += size

            # keep the while loop running for next run of for loop
            wind_start = timeframe_start
    #print("Windowed Data Created!")
    return x,y
#------------------------------------
# testing
if __name__ == "__main__":

    print("Trying create_windowed_data()!\n")
    content = create_windowed_data(["data2/video_android.mat"],
                                    0, timeframe_start = 0, timeframe_end= 600,size=100, shift=50)


