import numpy as np
import scipy.io as sio


# Reject a subject if a files trial number is less thatn 120 * %20 or %30
def create_windowed_data(filenames, timeframe_start, timeframe_end, size, min_trials,
                         channels_start=0, channels_end=60, trials_start=0, trials_end="end"):
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
        content = sio.loadmat(filename)
        index = [i for i, s in enumerate(list(content.keys())) if 'subj' in s]
        if not index:
            class_type = list(content.keys())[-1] + ''
        else:
            class_type = list(content.keys())[index[0]]
        data_x = np.asarray(content[class_type])
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

        # Check if trial number is below the minTrialLimit
        if data_x.shape[2] < min_trials:
            raise AssertionError("Subject has {} trials for agent {} which is less than minimum "
            "number of trials specified ({} trials).". format(label, data_x.shape[2], min_trials))

        if trials_end == "end":
            trials_end = data_x.shape[2]
        if channels_end > len(data_x):
            raise ValueError("More channels are requested than in the data")
        elif trials_end > len(data_x[0][0]):
            trials_end = len(data_x[0][0])

        # get all the information for each of the trials
        for trial in range(trials_start, trials_end):
            # for each of the window frames that you have decided to work on
            while wind_start + size <= timeframe_end:

                # select the part of data you want, 2d array to 1D with attachment one after the other.
                vector = data_x[channels_start:channels_end, wind_start: wind_start + size, trial]
                vector = vector.flatten()

                if vector.size != (size * (channels_end - channels_start)):
                    # TODO mantıklı bi error ver
                    raise ValueError("Ve")

                # add data for ml
                x.append(vector)
                y.append(label)

                # loop the while statement
                wind_start += size

            # keep the while loop running for next run of for loop
            wind_start = timeframe_start
    # print("Windowed Data Created!")
    return x, y
