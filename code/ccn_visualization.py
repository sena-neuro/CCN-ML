import matplotlib.pyplot as plt


def visualize(dir, name, sig_index, windows_val, vals):
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


def visualize_still_and_video(dir, name, v_sig_index, s_sig_index, windows_val, v_vals, s_vals):
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