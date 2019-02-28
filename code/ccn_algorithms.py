import numpy as np
import pandas as pd
import scipy.io as sio
import itertools
import ccn_io

# useful! Contains 3 for loops inside: files, trials, window_frames
def create_windowed_data(filenames,
    test_type=0, channels_start=0, channels_end=64, timeframe_start=100,
    timeframe_end=1100, trials_start=0, trials_end=50, step=100 ):
    """
        This method creates windowing for each of the channels put to it.
        It is the algorithm which allows comparison between the time trials. 
        No channel windowing present for the moment. 
    """

    try: 
        # output for usage
        x = []
        y = []

        # for each of the files
        for filename in filenames:

            # read the contents
            content    = ccn_io.read_file(filename)
            class_type = list(content.keys())[-1] + ''
            data_x     = np.asarray(content[class_type])[ channels_start : channels_end, timeframe_start : timeframe_end,  trials_start : trials_end]
            vector     = []
            window_frame = timeframe_start

            # get all the information for each of the trials
            for trial in range(trials_start, trials_end):
                # for each of the window frames that you have decided to work on
                while window_frame + step <= timeframe_end:

                    # select the part of data you want, 2d array to 1D with attachment one after the other.
                    vector = np.array( data_x[channels_start:channels_end,\
                                              window_frame: window_frame + step, trial])
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
                        elif class_type.find('robot') >= 0:
                            y.append(2)
                        else:
                            y.append(-1)
                    # loop the while statement
                    window_frame += step

                # keep the while loop running for next run of for loop
                window_frame = 0
        #print("Windowed Data Created!")
        return x,y

    except Exception:
        print("    Something went wrong with create_windowed_data()! Do not use until solved. To read whole message do not catch exception!\n")


# useful! Contains 3 for loops inside: files, trials, window_frames
def create_singly_windowed(filenames,
    test_type=0, channels_start=0, channels_end=20, timeframe_start=100,
    timeframe_end=1100, trials_start=0, trials_end=50, step=100 ):
    """
        This method creates windowing for each of the channels put to it.
        It is the algorithm which allows comparison between the time trials. 
        No channel windowing present for the moment. 
    """

    try: 
        # output for usage
        x = []
        y = []

        # for each of the files
        for filename in filenames:

            # read the contents
            content    = ccn_io.read_file(filename)
            class_type = list(content.keys())[-1] + ''
            data_x     = np.asarray(content[class_type])[ channels_start : channels_end, timeframe_start : timeframe_end,  trials_start : trials_end]
            vector     = []
            window_frame = timeframe_start

            # get all the information for each of the trials
            for trial in range(trials_start, trials_end):
                # for each of the window frames that you have decided to work on
                while window_frame + step <= timeframe_end:

                    # select the part of data you want, 2d array to 1D with attachment one after the other.
                    vector = np.array( data_x[channels_start:channels_end,\
                                              window_frame: window_frame + step, trial])

                    #vector = vector.flatten()
                    for case in vector:
                        # check if step is opening any problems
                        if case.size >= step:
                            # add data for ml
                            x.append(case.tolist())
                            # categorize for statistical process
                            if class_type.find('human') >= 0:
                                y.append(0)
                            elif class_type.find('android') >= 0:
                                y.append(1)
                            elif class_type.find('robot') >= 0:
                                y.append(2)
                            else:
                                y.append(-1)
                        # loop the while statement
                    window_frame += step

                # keep the while loop running for next run of for loop
                window_frame = 0
        #print("Windowed Data Created!")
        return x,y

    except Exception:
        print("    Something went wrong with create_singly_windowed()! Do not use until solved. To read whole message do not catch exception!\n")

#
#------------------------------------------------------------------
# Toy algorithm
def create_toy(data):
    """
        Toy example for the understanding of the algorithm needed for the process. 
    """
    x = []
    y = []
    class_type    = 0
    vector_list   = []
    vector        = []
    window_frame  = 0
    step          = 1
    timeframe_end = 2
    try: 
        for trial in range(0, 3):
            while( (window_frame+step) <= timeframe_end): 
                vector = np.array(data)[  0:2, window_frame:window_frame + step, trial ] # vectorize
                vector = vector.flatten() # from 2d to 1d for usage
                             
                # append the data for the x values too
                x.append(vector.tolist())
                y.append(0)
                
                window_frame += step
        return x,y
    except Exception: 
        print("    Something went wrong with the create_toy()!")
        
#------------------------------------
# testing
if __name__ == "__main__":

    print("Trying create_windowed_data()!\n")
    content = create_windowed_data(["data/android_motion_epochs.mat","data/robot_motion_epochs.mat"], 
                                    0, timeframe_start = 100, timeframe_end= 600,step=300)
    if content != None:
        print("    create_windowed_data() works.\n")

	# for toy example
    simple_data = [[[1,0,1],[0,1,0],[0,0,1]],[[2,0,0],[0,2,0],[0,0,2]],[[3,0,0],[0,3,0],[0,0,3]]]
    print("Trying create_toy()!\n")
    content = create_toy(simple_data)
    if content != None:
        print("    create_toy() works.\n")
