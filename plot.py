import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import argparse
import glob
import logging

plt.rcParams['text.usetex'] = True


def main_plot(list_of_data, smoothing_window=10,
              file_name='figure', labels=None, title="Reward Plot",
              x_label='Iterations',
              y_label='Rewards'):

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(28)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(20)
    axis_font = {'fontname':'Arial', 'size':'32'}

    # get a list of colors here.
    colors = sns.color_palette('colorblind', n_colors=len(list_of_data))
    rewards_smoothed = []

    for data, label, color in zip(list_of_data, labels, colors):
        episodes = np.arange(data.shape[0])
        smoothed_data = pd.DataFrame(data).rolling(smoothing_window, min_periods=smoothing_window).mean()

        rewards_smoothed.append(smoothed_data)
        data_mean = smoothed_data.mean(axis=1)
        data_std = smoothed_data.std(axis=1)
        ax.fill_between(episodes,  data_mean + data_std, data_mean - data_std, alpha=0.3,
                        edgecolor=color, facecolor=color)
        plt.plot(episodes, data_mean, color=color, linewidth=1.5,  label=label)

    ax.legend(loc='lower right', prop={'size' : 26})
    ax.set_xlabel(x_label,**axis_font)
    ax.set_ylabel(y_label, **axis_font)
    ax.set_title(title, **axis_font)

    fig.savefig('{}.pdf'.format(file_name))
    
    return fig


def get_paths(glob_path):
    return glob.glob(glob_path)

def load_and_stack_npy(glob_path):
    path_to_npys = get_paths(glob_path)
    if len(path_to_npys) == 0:
        raise ValueError('No paths found')

    datas = []

    # here i am truncating the runs since some of the experiments did not run until completion
    min_timesteps = np.Inf
    for path in path_to_npys:
        datas.append(np.load(path))
        if datas[-1].shape[0] < min_timesteps:
            logging.warning('Truncating to shorter run')
            min_timesteps = datas[-1].shape[0]

    truncated_data = []
    for data in datas:
        truncated_data.append(data[:min_timesteps])
    datas = np.stack(truncated_data, 1)

    return datas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", help="Glob paths to the folder with data", nargs='+')
    parser.add_argument('--labels', default=[], nargs='+')
    parser.add_argument('--title', default='Reward Plot')
    parser.add_argument('--xlabel', default='Episodes (1eps = 5000 timesteps)')
    parser.add_argument('--ylabel', default='Rewards')
    parser.add_argument('--smoothing_window', default=10, type=int)

    args = parser.parse_args()

    if len(args.labels) < len(args.paths):
        args.labels.extend([''] * (len(args.paths)-len(args.labels)))

    print('Number of paths provided: {}'.format(len(args.paths)))
    datas = []
    for path in args.paths:
        datas.append(load_and_stack_npy(path))
        print('Number of replicates loaded from {}: {}'.format(path, datas[-1].shape))



    main_plot(datas,
              smoothing_window=args.smoothing_window,
              file_name=args.title.replace(' ', ''),
              labels=args.labels,
              title=args.title,
              x_label=args.xlabel,
              y_label=args.ylabel)

if __name__ == '__main__':
    main()