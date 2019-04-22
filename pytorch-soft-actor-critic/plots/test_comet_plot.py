import argparse
import csv
import json
import os

from comet_ml import API
import comet_ml
import numpy as np
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

    title = rows.get('filename')[0]
    xlabel = rows.get('xlabel')[0]
    ylabel = rows.get('ylabel')[0]
    metric = rows.get('metric')[0]
    data = {key: value for key, value in rows.items() if 'experiment' in key.lower()}

    return title, xlabel, ylabel, metric, data


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


def get_data(source_filename):
    title, xlabel, ylabel, metric, data = extract_excel_data(source_filename)
    comet_api, comet_username, comet_project = connect_to_comet()

    for exp_name, experiment in data.items():
        if len(experiment) > 0:
            for exp_key in experiment:
                raw_data = comet_api.get("%s/%s/%s" %(comet_username, comet_project, exp_key))
                print('stop')




def main(args):
    source_filename = args.source_filename

    get_data(source_filename)
    # datas, labels, title, xlable, ylabel = get_data()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_filename', default='plot_source.csv')
    args = parser.parse_args()

    main(args)
