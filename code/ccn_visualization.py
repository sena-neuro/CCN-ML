# plotting
from textwrap import wrap

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from itertools import compress


def plot_confusion_matrix(cm, classes,
                          f_name,
                          normalize=False,
                          title=None,
                          avg=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = np.asarray(cm)
    if normalize:
        title += ", normalized"
    else:
        title += ", without normalization"

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

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

    plt.savefig(f_name, bbox_inches='tight')
    plt.clf()
    plt.close()


def visualize(f_name, sig_index, windows, avg_values, chance_level):
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
    ax.axhline(y=chance_level, color='red', alpha=0.5, label='chance level')

    ax.legend(loc='upper right')
    ax.set_title('Classification accuracies')

    fig.savefig(f_name, bbox_inches='tight')
    plt.clf()
    plt.close()

# TODO clean up visualization arguments
def visualize_two_curves(f_name, first_sig_index, second_sig_index, windows_val, first_avg, second_avg, chance_level,
                         first_sems, second_sems, first_exp_params, second_exp_params, error_bar=True):

    if first_exp_params['input_type'] == second_exp_params['input_type']:
        # Will be used for plot title
        left_title = "Input type: " + first_exp_params['input_type']

        label1 = first_exp_params['exp_type']
        label2 = second_exp_params['exp_type']
    elif first_exp_params['exp_type'] == second_exp_params['exp_type']:
        # Will be used for plot title
        left_title = "Experiment type: " + first_exp_params['exp_type']

        label1 = first_exp_params['input_type']
        label2 = second_exp_params['input_type']

    # take average accuracies from avg list
    # avg lists are in the form : [[time_window, accuracy]]
    # so 2-dim lists where in the inner list
    # first element is the time window (in a string)
    # and the second element is the accuracy
    first_vals = [x[-1] for x in first_avg]
    second_vals = [x[-1] for x in second_avg]

    fig, ax = plt.subplots()
    if (error_bar):
        # Video error bars
        ax.errorbar(x=windows_val, y=first_vals, yerr=first_sems, capsize=5, label=label1)

        # Still error bars
        ax.errorbar(x=windows_val, y=second_vals, yerr=second_sems, capsize=5, label=label2)
    else:
        # Without errorbars
        ax.plot(windows_val, first_vals, label=label1)
        ax.plot(windows_val, second_vals, label=label2)

    # Mark significant time windows with red dots
    ax.plot(list(compress(windows_val, first_sig_index)), list(compress(first_vals, first_sig_index)),
            linestyle="none", color='r', marker='o')
    ax.plot(list(compress(windows_val, second_sig_index)), list(compress(second_vals, second_sig_index)),
            linestyle="none", color='r', marker='o')

    # show baseline and chance level
    ax.axvline(x=0, color='black', alpha=0.5, linestyle='--', label='end of baseline period')
    ax.axhline(y=chance_level, color='red', alpha=0.5, label='chance level')

    if ('hra' == first_exp_params['target_labels']):
        midtitle = "Classification among Human, Robot and Android agents"
    elif ('hr' == first_exp_params['target_labels']):
        midtitle = "Classification between Human and Robot agents"
    elif ('ra' in first_exp_params['target_labels']):
        midtitle = "Classification between Robot and Android agents"
    elif ('ah' in first_exp_params['target_labels']):
        midtitle = "Classification between Human and Android agents"

    ax.set_xlabel('time (ms)')
    ax.set_ylabel('classification accuracy')

    ax.legend(loc='upper right')
    right_title= "Window size=" + str(first_exp_params['w_size']) + "ms, window shift=" + str(first_exp_params['shift']) + "ms"
    fig.suptitle(midtitle,fontsize=12, fontweight='bold')
    ax.set_title("\n".join(wrap(left_title)), fontsize=9,loc='left')
    ax.set_title("\n".join(wrap(right_title)), fontsize=9, loc='right' )

    # the window size and shift from the data
    fig.savefig(f_name, bbox_inches='tight')
    plt.clf()
    plt.close()
