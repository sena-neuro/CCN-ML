# plotting
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          avg=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = np.asarray(cm)
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize or avg else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, ax


def visualize(dir, name, sig_index, windows, avg_values):
    vals = [x[-1] for x in avg_values]

    # decide on time
    windows_val = [2 * (x - 100) for x in [int(wind_frame.strip('()').split(',')[0])
                                           for wind_frame in windows]]

    # take average accuracies from avg list
    # avg lists are in the form : [[time_window, accuracy]]
    # so 2-dim lists where in the inner list
    # first element is the time window (in a string)
    # and the second element is the accuracy
    vals = [x[-1] for x in avg_values]

    fig, ax = plt.subplots()
    ax.plot(windows_val, vals)

    ax.plot([windows_val[x] for x in sig_index], [vals[x] for x in sig_index],
            linestyle='none', color='r', marker='o')

    # show starting understanding and chance level
    ax.axvline(x=0, color='black', alpha=0.5, linestyle='--', label='end of baseline period')
    ax.axhline(y=0.33, color='red', alpha=0.5, label='chance level')

    ax.legend(loc='upper right')
    ax.set_title('Classification accuracies')

    fig.savefig(dir + name, bbox_inches='tight')
    plt.clf()
    plt.close()


def visualize_still_and_video(dir, name, v_sig_index, s_sig_index, windows, v_avg, s_avg):
    # decide on time
    windows_val = [2 * (x - 100) for x in [int(wind_frame.strip('()').split(',')[0])
                                           for wind_frame in windows]]

    # take average accuracies from avg list
    # avg lists are in the form : [[time_window, accuracy]]
    # so 2-dim lists where in the inner list
    # first element is the time window (in a string)
    # and the second element is the accuracy
    v_vals = [x[-1] for x in v_avg]
    s_vals = [x[-1] for x in s_avg]

    fig, ax = plt.subplots()
    ax.plot(windows_val, v_vals, 'g', label="Video")
    ax.plot(windows_val, s_vals, 'b', label='Still')

    # print(windows_val)
    ax.plot([windows_val[x] for x in v_sig_index], [v_vals[x] for x in v_sig_index],
            linestyle="none", color='r', marker='o')
    ax.plot([windows_val[x] for x in s_sig_index], [s_vals[x] for x in s_sig_index],
            linestyle="none", color='r', marker='o')

    # show starting understanding and chance level
    ax.axvline(x=0, color='black', alpha=0.5, linestyle='--', label='end of baseline period')
    ax.axhline(y=0.33, color='red', alpha=0.5, label='chance level')

    ax.legend(loc='upper right')
    ax.set_title('Classification accuracies')  # If we really want to, we can get
    # the window size and shift from the data

    fig.savefig(dir + name, bbox_inches='tight')
    plt.clf()
    plt.close()