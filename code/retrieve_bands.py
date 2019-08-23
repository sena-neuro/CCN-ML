#!/usr/bin/env python
# coding: utf-8

# * https://raphaelvallat.com/bandpower.html (for computing average bandpower of EEG Signal)
# * https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.welch.html ( welch method of FT)
# * https://en.wikipedia.org/wiki/Electroencephalography (for the bands and what they mean)


import scipy.io as sio
# Libraries needed
from numpy import *
from numpy.fft import *
# from matplotlib import *
# from scipy import *
from pylab import linspace
from scipy.signal import freqz


def get_bands(data, low =0, high = 4):
    """ Take alpha, beta, gamma or delta bands depending on input.
   
    Parameters
    ----------
    data : 3d-array
        Input signal of the experiment.

    Return
    ------
    filtered : 3d-array
        Filtered band for the signal frequency.
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import signal

    # sampling frequency
    sf  = 500
    win = 1050
    
    # Define delta lower and upper limits
    i = 0
    for each_channel in range(0,data.shape[0]):
        for each_trial in range(0,data.shape[2]):
            data_signal = data[each_channel,:,each_trial]
            
            time = np.arange(data_signal.size) / sf
            freqs, psd = signal.welch(data_signal, sf, nperseg=win)
            
            if i < 5:
                # Find intersecting values in frequency vector
                idx_delta = np.logical_and(freqs >= low, freqs <= high)

                # Plot the power spectral density and fill the delta area
                plt.figure(figsize=(7, 4))
                plt.plot(freqs, psd, lw=2, color='k')
                plt.fill_between(freqs, psd, where=idx_delta, color='skyblue')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Power spectral density (uV^2 / Hz)')
                plt.xlim([0, 10])
                plt.ylim([0, psd.max() * 1.1])
                plt.title("Welch's periodogram")
                sns.despine()
            i += 1

def retrieve_bands( data_signal, fs = 500.0 ):
    """Take from the signal only needed bands."""
    
    #<========================= Parameters ======================================>
    y   = data_signal
    L   = len(y)            # signal length
    T   = 1/fs              # sample time
    t   = linspace(1,L,L)*T  # time vector

    M   = 512      # Set number of weights as 128
    f   = fs*linspace(0,int(L/10),int(L/10))/L  # single side frequency vector, real frequency up to fs/2
    Y   = fft(y)

    #figure()
    filtered = []
    b        = [] # store filter coefficient
    
    #<========================= Filter  ==========================================>
    cutoff   = [0.5,4.0,7.0,12.0,30.0]
    for band in range(0, len(cutoff)-1):
        wl = 2 * cutoff[ band]/ fs*pi
        wh = 2 * cutoff[ band +1]/ fs*pi
        bn = zeros(M)

        for i in range(0,M):     # Generate bandpass weighting function
            n = i-M/2            # Make symmetrical
            if n == 0:
                bn[i] = wh/pi - wl/pi
            else:
                bn[i] = (sin(wh*n))/(pi*n) - (sin(wl*n))/(pi*n)   # Filter impulse response

        bn = bn*kaiser(M,5.2)  # apply Kaiser window, alpha= 5.2
        b.append(bn)

        [w,h]=freqz(bn,1)
        filtered.append(convolve(bn, y)) # filter the signal by convolving the signal with filter coefficients
    
    """
    #<========================= Plot =============================================>
  
    figure(figsize=[16, 10])
    subplot(2, 1, 1)
    plot(y)
    for i in range(0, len(filtered)):
        y_p = filtered[i]
        plot(y_p[ int(M/2):int(L+M/2)])
    axis('tight')
    title('Time domain')
    xlabel('Time (seconds)')

    subplot(2, 1, 2)
    plot(f,2*abs(Y[0:int(L/10)]))
    for i in range(0, len(filtered)):
        Y = filtered[i]
        Y = fft(Y [ int(M/2):int(L+M/2)])
        plot(f,abs(Y[0:int(L/10)]))
    axis('tight')
    legend(['original','delta band, 0-4 Hz','theta band, 4-7 Hz','alpha band, 7-12 Hz','beta band, 12-30 Hz'])

    for i in range(0, len(filtered)):   # plot filter's frequency response
        H = abs(fft(b[i], L))
        H = H*1.2*(max(Y)/max(H))
        plot(f, 3*H[0:int(L/10)], 'k')    
    axis('tight')
    title('Frequency domain')
    xlabel('Frequency (Hz)')
    subplots_adjust(left=0.04, bottom=0.04, right=0.99, top=0.97)
    savefig('filtered.png')
    """
    return filtered

# testing
if __name__ == "__main__":
    #list_starter = ['subj01_prior_video_robot.mat','subj01_prior_video_human.mat']#,'subj01_prior_video_android.mat']
    #for name in list_starter:
    name = 'subj01_prior_video_human.mat' 
    key  = name[:-4]

    data_array = sio.loadmat(name)[key]
    info_delta = []
    info_theta = []
    info_alpha = []
    info_beta  = []
    import time

    start = time.time()

    """
    print(data_array.shape)
    data_trial = data_array[0,:,0]

    result = retrieve_bands(data_trial)
    print(data_trial.shape)
    print(result[1].shape)

    """
    for each_trial in range(0, data_array.shape[2]):

        # info for each trial
        trial_delta = []
        trial_theta = []
        trial_alpha = []
        trial_beta  = []

        for each_channel in range(0, data_array.shape[0]):
            result = retrieve_bands(data_array[each_channel, :, each_trial])

            trial_delta.append(result[0].tolist())
            trial_theta.append(result[1].tolist())
            trial_alpha.append(result[2].tolist())
            trial_beta.append(result[3].tolist())

        info_delta.append(trial_delta)
        info_theta.append(trial_theta)
        info_alpha.append(trial_alpha)
        info_beta.append(  trial_beta)

    sio.savemat(key + '_delta.mat', {key : np.rollaxis( np.array( info_delta), 0, 3)})
    sio.savemat(key + '_theta.mat', {key : np.rollaxis( np.array( info_theta), 0, 3)})
    sio.savemat(key + '_alpha.mat', {key : np.rollaxis( np.array( info_alpha), 0, 3)})
    sio.savemat(key +  '_beta.mat', {key:np.rollaxis(np.array( info_beta),0,3)})

    print(key)

    print("The whole process is ",time.time()-start)