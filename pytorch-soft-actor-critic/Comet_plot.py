from comet_ml import API
import comet_ml
import numpy as np

import csv
import pandas as pd
import argparse
import logging
import os
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper', font_scale=1.3)
sns.set_style('whitegrid')
sns.set_palette('colorblind')
plt.rcParams['text.usetex'] = True



def getData():
    filename = "plot_source.xls"
    rows = []
    with open(filename, 'r') as csvfile:
        # Creating a csv reader object
        csvreader  = csv.reader(csvfile)

        for row in csvreader:
            rows.append(row)

    data_row = []
    data_source = []
    labels = []

    for it, row in enumerate(rows):
        for col in row:
            data_row.append(col.split('\t'))


    for it, row in enumerate(data_row):
        #print(row[0])
        if row[0]=='filename':
            title = row[1]
            continue
        if row[0]=='xlabel':
            xlabel = row[1]
            continue
        if row[0]=='ylabel':
            ylabel = row[1]
            continue
        if row[0]=='metric':
            metric = row[1]
            continue
        labels.append(row[0])
        data_row = []
        for col in row:
            data_row.append(col)
        data_source.append(data_row)


    # Setting up Comet:

    if os.path.isfile("settings.json"):
        with open('settings.json') as f:
            data = json.load(f)
        comet_apikey = data["apikey"]
        comet_username = data["username"]
        comet_restapikey = data["restapikey"]
        comet_project = data["project"]

    print("COMET_REST_API_KEY=%s" %(comet_restapikey))
    with open('.env', 'w') as writer:
        #print("\ntrying to write!!")
        writer.write("COMET_API_KEY=%s\n" %(comet_apikey))
        writer.write("COMET_REST_API_KEY=%s\n" %(comet_restapikey))

    # fd = os.open(".env", os.O_RDWR|os.O_CREAT )
    # _ = os.write(fd, "COMET_API_KEY=%s" %(comet_apikey))
    # _ = os.write(fd, "COMET_REST_API_KEY=%s" %(comet_restapikey))
    # os.close(fd)

    comet_api = API()

    datas = []

    data_temp = []
    truncate_value = np.inf
    for _, line in enumerate(data_source):
        data_stream = []
        for idx,val in enumerate(line):
            if idx == 0:
                continue
            #print("%s/%s/%s" %(comet_username, comet_project, val))
            if val == '':
                continue
            RawData = comet_api.get("%s/%s/%s" %(comet_username, comet_project, val))
            #import ipdb; ipdb.set_trace()
            data_stream.append([x[1] for x in RawData.metrics_raw[metric]])

        lengths = []
        for data in data_stream:
            lengths.append( len(data) )

        smallest = min(lengths)
        truncate_value = min(truncate_value, smallest)
        for i, data in enumerate(data_stream):
            data_stream[i] = data[:smallest]

            # Now stack the data_Stream and append it to datas.
        data_temp.append(np.stack(data_stream,1))


    #truncated_data = []
    for idx, data in enumerate(data_temp):
        data_temp[idx] = data_temp[idx][:truncate_value]

    for data in data_temp:
        datas.append(data)

    if title=='' or xlabel=='' or ylabel=='':
        # Handle Defaults
        print("Error in reading CSV file. Ensure filename, x and y labels are present")
    #import ipdb; ipdb.set_trace()
    return datas, labels, title, xlabel, ylabel


# Generalized plotting function
def main_plot(list_of_data, smoothing_window=10,
              file_name='figure', labels=None, title="Reward Plot",
              x_label='Iterations',
              y_label='Rewards'):

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot()
    for label in (ax.get_xticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(28)
    for label in (ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(28)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(20)
    axis_font = {'fontname':'Arial', 'size':'32'}

    # get a list of colors here.
    colors = sns.color_palette('colorblind', n_colors=len(list_of_data))
    #colors = sns.color_palette('cubehelix', n_colors=len(list_of_data))
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

    fig.savefig('{}.png'.format(file_name))
    fig.savefig('{}.pdf'.format(file_name))

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoothing_window', default=10, type=int)
    args = parser.parse_args()
    print("Fetching the data")

    datas, labels, title, xlabel, ylabel = getData()

    main_plot(datas,smoothing_window=args.smoothing_window,file_name=title,labels=labels,title=title,x_label=xlabel,y_label=ylabel)


if __name__ == '__main__':
    main()
