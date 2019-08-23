#!/usr/bin/env python
# coding: utf-8

# Import needed information 
import mne, os, glob
from mne.io import read_raw_brainvision, read_epochs_eeglab
import scipy
import numpy as np
# Get the files.
# Print Information.
# Store needed information.
directory = '/home/user/Desktop/EEG_data/Static/'
for each in glob.glob(directory + "*/*.set"): 
  
    subj = each.replace(directory,"")
    # read epochs object
    eeglab_obj = read_epochs_eeglab(each)
    
    # apply basealine
    eeglab_obj.apply_baseline((-0.2,0))

    # remove channels
    eeglab_obj.pick_channels([ "Fz", "F3", "F7" , "FT9" , "FC5", "FC1", "C3", "T7", "TP9" , "CP5",\
                             "CP1", "Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8", "TP10",\
                             "CP6", "CP2", "Cz", "C4", "T8", "FT10", "FC6", "FC2", "F4", "F8",\
                             "AF7" ,"AF3", "AFz"  , "F1","F5" , "FT7", "FC3" ,"FCz" , "C1", "C5", \
                             "TP7", "CP3" ,"P1" , "P5", "PO7", "PO3", "POz", "PO4", "PO8", "P6", \
                             "P2", "CPz", "CP4", "TP8", "C6", "C2", "FC4", "FT8", "F6", "F2", "AF4", "AF8"])
    # visualize
    #fig  = eeglab_obj.plot_image()
    # None
    
    # store for the project
    data = eeglab_obj.get_data()
    # roll axis in order to settle down to previous layout of problem
    rolled_data = np.rollaxis(data, 2)
    rolled_data = np.rollaxis(rolled_data, 2)
    
    name = subj[7:].replace("set","mat")
    each.replace("set","mat")
    scipy.io.savemat( each.replace("set","mat"), { name.replace(".mat","").replace("_s_","_static_"): rolled_data})