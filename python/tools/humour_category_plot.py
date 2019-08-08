#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from scipy.stats import pearsonr, spearmanr


def create_plot(experiment_file, task):
    categories = {'hetpun': 'purple', 'hompun': 'blue', 'nonpun': 'orange', 'non': 'black'}

    if task == 'metaphor':
        pass
    elif task == 'humour':
        with open(experiment_file, 'r') as f:
            reader = csv.DictReader(f)
            bws, gppl, color = [], [] ,[]
            for row in reader:
                bws.append(float(row['bws']))
                gppl.append(float(row['predicted']))
                color.append(categories[row['category']])
        spearman = spearmanr(bws, gppl)
        print('Spearman:', spearman)
        print('')

        # create plot
        plt.ioff()
        plt.scatter(bws, gppl, color)
        plt.savefig(experiment_file + '_cats.png')
        plt.close()
    else:
        raise ValueError('task needs to be one of [metaphor, humor]')


create_plot(sys.argv[1], sys.argv[2])
