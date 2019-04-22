import argparse
import csv
import json
import os

from comet_ml import API
import numpy as np
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
import seaborn as sns

# Set plotting style
sns.set_context('paper', font_scale=1.3)
sns.set_style('whitegrid')
sns.set_palette('colorblind')
plt.rcParams['text.usetex'] = True


def extract_excel_data(source_filename):
    with open(source_filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        rows = {row[0]: row[1:] for row in csvreader}

    labels = {}
    labels['title'] = rows.get('filename')[0]
    labels['x_label'] = rows.get('xlabel')[0]
    labels['y_label'] = rows.get('ylabel')[0]
    labels['metric'] = rows.get('metric')[0]

    data = {key: value for key, value in rows.items() if 'experiment' in key.lower()}
    labels['experiments'] = [key.split(':')[1] for key, value in data.items()]

    return labels, data


def connect_to_comet():
    if os.path.isfile("settings.json"):
        with open("settings.json") as f:
            keys = json.load(f)
            comet_apikey = keys.get("apikey")
            comet_username = keys.get("username")
            comet_restapikey = keys.get("restapikey")
            comet_project = keys.get("project")

    print("COMET_REST_API_KEY=%s" %(comet_restapikey))
    with open('.env', 'w') as writer:
        writer.write("COMET_API_KEY=%s\n" %(comet_apikey))
        writer.write("COMET_REST_API_KEY=%s\n" %(comet_restapikey))

    comet_api = API()
    return comet_api, comet_username, comet_project


def truncate_runs(data_runs):
    last_data_points = [run[-1] for run in data_runs]
    run_end_times = [timestep for timestep, value in last_data_points]
    earliest_end_time = min(run_end_times)

    clean_data_runs = []
    for run in data_runs:
        clean_data_runs.append([(x, y) for x, y in run if x <= earliest_end_time])

    return clean_data_runs


def get_data(title, x_label, y_label, metric, data):
    if not title or not x_label or not y_label or not metric:
        print("Error in reading CSV file. Ensure filename, x and y labels, and metric are present.")
        exit(1)

    comet_api, comet_username, comet_project = connect_to_comet()

    # Accumulate data for all experiments.
    data_experiments = []
    for exp_name, runs in data.items():
        # Accumulate data for all runs of a given experiment.
        data_runs = []
        if len(runs) > 0:
            for exp_key in runs:
                raw_data = comet_api.get("%s/%s/%s" %(comet_username, comet_project, exp_key))
                data_points = raw_data.metrics_raw[metric]
                data_runs.append(data_points)

            clean_data_runs = truncate_runs(data_runs)
            data_experiments.append(clean_data_runs)

    return data_experiments


def plot(**kwargs):
    labels = kwargs.get('labels')
    data = kwargs.get('data')

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot()
    for label in (ax.get_xticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(28)
    for label in (ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(28)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize(20)
    axis_font = {'fontname': 'Arial', 'size': '32'}
    colors = sns.color_palette('colorblind', n_colors=len(data))

    rewards = []
    for runs, label, color in zip(data, labels.get('experiments'), colors):
        plt.plot(runs[:, 0], runs[:, 1], color=color, linewidth=1.5, label=label)
        print('hello')

        # data_experiments = np.array(data)
        # episodes = np.arange(data_experiments.shape[0])
        # cleaned_data = pd.DataFrame(data_experiments)
        # rewards.append(cleaned_data)
        #
        # data_experiments_mean = cleaned_data.mean(axis=1)
        # data_experiments_std = cleaned_data.std(axis=1)
        # ax.fill_between(episodes, data_experiments_mean + data_experiments_std, data_experiments_mean - data_experiments_std,
        #                 alpha=0.3, edgecolor=color, facecolor=color)
        # plt.plot(episodes, data_experiments_mean, color=color, linewidth=1.5, label=label)

    ax.legend(loc='lower right', prop={'size': 26})
    ax.set_xlabel(labels.get('x_label'), **axis_font)
    ax.set_ylabel(labels.get('y_label'), **axis_font)
    fig.subplots_adjust(bottom=0.2)
    fig.subplots_adjust(left=0.2)
    ax.set_title(labels.get('title'), **axis_font)

    fig.savefig('../install/{}.pdf'.format(labels.get('title')))

    return


def main(args):
    source_filename = args.source_filename

    labels, data = extract_excel_data(source_filename)
    data_experiments = get_data(labels.get('title'), labels.get('x_label'), labels.get('y_label'), labels.get('metric')
                                , data)
    plot(labels=labels, data=data_experiments)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_filename', default='plot_source.csv')
    args = parser.parse_args()

    main(args)
