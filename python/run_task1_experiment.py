'''
Run task 1: no features, use all available training data, see if the GPPL model produces a similar ranking to BWS.

'''
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s')
import sys
# include the paths for the other directories
sys.path.append("./python")
sys.path.append("./python/models")

from gp_pref_learning import GPPrefLearning
from collections import namedtuple, OrderedDict
from datetime import datetime
from nltk.tokenize import word_tokenize
from operator import itemgetter
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import ParameterGrid
from vuamc import Vuamc

import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import re

# data paths
humour_path = './data/pl-humor-full/results.tsv'
metaphor_path = './data/vuamc_crowd/all.csv'
vuamc_path = './data/VU Amsterdam Metaphor Corpus/2541/VUAMC_with_novelty_scores.xml'

# output data path
overview_file = 'overview.csv'

# set numpy seed
rnd_seed = 41
np_seed = 1337

def get_instance(task, instance, vuamc):
    """
    Get the instance corresponding to a given id, depending on the task.
    For metaphor, instance is and index id to retrieve data from the VUAMC.
    For humour, instance is a sentence

    Needs a global map index->instance_id.
    """
    if task == 'metaphor':
        _, sentence, metaphor = vuamc.get(instance)
#        tokens = [t.covered_text.lower() for t in sentence.tokens]
#            if t.covered_text.lower() not in embeddings and t.lemma.lower() in embeddings:
#                    tokens.append(t.lemma.lower())
#                else:
#                    tokens.append(t.covered_text.lower())
#        else:
#            for t in instance:
#                tokens.append(t.lower())
        focus_position = int(instance.split('.')[2])  # 0-based index of the metaphor token in the sentence
        return [t.covered_text.lower() for t in sentence.tokens], focus_position
    elif task == 'humour':
        return [t.lower() for t in instance], None
    else:
        raise ValueError('task must be in [metaphor, humour]')


def load_crowd_data_TM(path):
    """
    Read csv and create preference pairs of tokenized sentences.

    :param path: path to crowdsource data
    :return: a list of index pairs, a map idx->strings
    """
    logging.info('Loading crowd data...')

    pairs = []
    idx_instance_list = []
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # skip header row
        for line_no, line in enumerate(reader):
            answer = line[1]
            A = word_tokenize(line[2])
            B = word_tokenize(line[3])
            # add instances to list (if not alreay in it)
            if A not in idx_instance_list:
                idx_instance_list.append(A)
            if B not in idx_instance_list:
                idx_instance_list.append(B)
            # add pairs to list (in decreasing preference order)
            if answer == 'A':
                pairs.append((idx_instance_list.index(A), idx_instance_list.index(B)))
            if answer == 'B':
                pairs.append((idx_instance_list.index(B), idx_instance_list.index(A)))
    return pairs, idx_instance_list


def load_crowd_data_ED(path):
    """
    Read csv and create preference pairs of VUAMC ids representing sentences with focus.

    :param path: path to crowdsource data
    :return: a list of index pairs, a map idx->vuamc_id
    """
    logging.info('Loading crowd data...')

    pairs = []
    idx_instance_list = []

    skipped = 0
    corrected = [0]
    # create 3*5 pairs from each line:
    # - +3: a = best > b, c, d
    # - +2: d = worst < b, c
    # - *3, because a HIT contains 3 comparisons
    Hit = namedtuple('Hit', 'ids nov_id con_id')

    def correct(id_map, bws_id):
        try:
            if bws_id.startswith('$'):
                # print('  Corrected bws id', bws_id)
                corrected[0] += 1
                return id_map[bws_id[1:]]
            return id_map[bws_id]
        except KeyError:
            return ''

    if os.path.isfile(path):
        paths = [path]
    elif os.path.isdir(path):
        paths = [os.path.join(path, fname) for fname in os.listdir(path)]

    for path in paths:
      logging.info('Loading crowd data from %s...', path)
      with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for line_no, line in enumerate(reader):
            for i in range(0,3):  # 3 assignments per HIT
                # add instances to list (if not alreay in it)
                bws_vuamc_ids = OrderedDict(zip(line[27+(i*16):27+(i*16) + 4], line[27+(i*16)+4:27+(i*16) + 4 + 4]))
                con_id = bws_vuamc_ids[line[87+(i*2)]] if line[87+(i*2)] in bws_vuamc_ids else None
                nov_id = bws_vuamc_ids[line[88+(i*2)]] if line[88+(i*2)] in bws_vuamc_ids else None
                # check hit validity
                if nov_id is None or con_id is None or None in [re.match(r'^\d+\.\d+\.\d+$',vuamc_id) for vuamc_id in bws_vuamc_ids.values()]:
#                    logging.warn('Skipping corrupt VUAMC id in line [%s]', line_no)
                    continue
                for vuamc_id in bws_vuamc_ids.values():
                    if vuamc_id not in idx_instance_list:
                        idx_instance_list.append(vuamc_id)
                if OPTIONS['data_mode'] == 'pairs':
                    pairs.append((idx_instance_list.index(nov_id), idx_instance_list.index(con_id)))
                else:
                    for current_id in bws_vuamc_ids.values():
                        if current_id != nov_id:  # best (a) > b, c, d
                            pairs.append((idx_instance_list.index(nov_id), idx_instance_list.index(current_id)))  # append "novel" pairs
                            if current_id != con_id:  # worst (d) < b, c
                                pairs.append((idx_instance_list.index(current_id), idx_instance_list.index(con_id)))  # append "conventionalized" pairs
    return pairs, idx_instance_list


def extract_features(requested_idxs):
    logging.warn('Using diagonal covariance matrix without feature vectors!')
    feature_vectors = np.array(requested_idxs)
    feature_vectors.shape += (1,)
    return feature_vectors


def train(train_split, train_idxs, idx_instance_list, task):
    """
    Train a model using the given training set.

    :param traing_split: list of training examples (index-pairs)
    :param idx_vuamc_map: 
    :param embeddings: 
    :param vuamc: 
    """
    logging.info('Training model...')

    items_feat = extract_features(train_idxs)  # needs to be mapped to the ids used in the training_splits
    a1_train, a2_train = zip(*train_split)

    # re-assign indexes based on position in the index list (this makes a*_train compatible with the order of items_feat)
    a1_train = [train_idxs.index(idx) for idx in a1_train]
    a2_train = [train_idxs.index(idx) for idx in a2_train]

#    ls_initial = compute_median_lengthscales(items_feat)
#    rate_ls=1.0 / np.mean(ls_initial),
    ls_initial = None  # default
    rate_ls = 10  # default

    # following ES' advice here
    ninducing = min(len(train_idxs), 400)

    model = GPPrefLearning(ninput_features=items_feat.shape[1],
                           kernel_func='diagonal',
                           ls_initial=ls_initial,
                           verbose=True,
                           shape_s0=2.0,
                           rate_s0=200.0,
                           rate_ls=rate_ls,
                           use_svi=True,
                           ninducing=ninducing,
                           max_update_size=1000,
                           kernel_combination='*',
                           forgetting_rate=0.9,
                           delay=1.0)

    model.max_iter_VB = 2000

    #model.max_iter_G = 100
    #model.fixed_s = True if 'debug' in OPTIONS else False
    # in the data loading step we already sorted each pair, preference-descending wise
    # i.e. the first instance is always the preferred (more funny, more novel) one
    prefs_train = [1] * len(a1_train)  # 1: a1 preferred over (i.e. more novel/funny than) a2, using input_type='binary'

    logging.info("**** Started training GPPL ****")
    print("items_feat", items_feat.shape)
    print('a1_train', np.array(a1_train).shape, 'a2_train', np.array(a2_train).shape, 'prefs_train', np.array(prefs_train).shape)
    model.fit(np.array(a1_train), np.array(a2_train), items_feat, np.array(prefs_train, dtype=float), optimize=OPTIONS['optimization'], input_type='binary')
    logging.info("**** Completed training GPPL ****")

    return model


def save_model(model, experiment_dir, output_filename):
    """Save a given model to pkl."""
    pkl_file = os.path.join(experiment_dir, output_filename + '.model')
    logging.info('Saving model to %s', pkl_file)
    with open(pkl_file, 'wb') as f:
        pickle.dump(model, f)


def load_model(path):
    """Load a model from a pkl file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def run(task, experiment_dir):
    """
    Run a given configuration.
    """
    logging.info('Running GPPL on task [%s] for task 1', task)
    output_filename = 'results-' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    start_time = datetime.now()

    if task == 'humour':
        pairs, idx_instance_list = load_crowd_data_TM(humour_path)
        vuamc = None
    elif task == 'metaphor':
        pairs, idx_instance_list = load_crowd_data_ED(metaphor_path)
        vuamc = Vuamc(vuamc_path)

    logging.warn('Training split is 100%, using complete set as dev set (emulating BWS data availability)...')
    train_idxs = list(range(len(idx_instance_list)))
    dev_idxs = train_idxs
    train_split = pairs[:10]

    # train model on cut
    model = train(train_split, train_idxs, idx_instance_list, task)
    save_model(model, experiment_dir, output_filename)

    # apply model to dev split
    dev_feats = extract_features(dev_idxs)
    predicted_f, _ = model.predict_f(out_feats=dev_feats)
    predicted = dict(zip(dev_idxs, [float(p) for p in predicted_f]))

    rows = []
    gold_v = []
    gppl_v = []

    if task == 'metaphor':
        for i, (dev_idx, pred) in enumerate(predicted.items()):
            vuamc_id = idx_instance_list[dev_idx]
            _, sentence, metaphor = vuamc.get(vuamc_id)
            rows.append([vuamc_id, metaphor.score, pred, metaphor.covered_text, sentence.pp()])
            try:
                if metaphor.score is not None and pred is not None:
                   gold_v.append(float(metaphor.score))
                   gppl_v.append(pred)
            except Exception as e:
                print("S:", sentence.pp())
                print("M:", metaphor)
                #print("V:", vuamc_ids[i])
                print("N:", metaphor.score)
                print("P:", predicted_f[i][0])
                raise e

        # write output
#        rows = sorted(rows, key=itemgetter(1), reverse=True)  # sort rows by gold
        with open(os.path.join(experiment_dir, output_filename + '.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'novelty', 'predicted', 'metaphor', 'sentence'])
            writer.writerows(rows)
    else:
        scores = dict()
        with open('./data/pl-humor-full/item_scores.txt', 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                scores[' '.join(word_tokenize(line[2]))] = float(line[1])
        for dev_idx, pred in predicted.items():
            instance = idx_instance_list[dev_idx]
            bws = scores[' '.join(instance)]  # get gold score
            gold_v.append(bws)
            gppl_v.append(pred)
            rows.append([dev_idx, bws, pred, ' '.join(instance)])

        # write output
        rows = sorted(rows, key=itemgetter(1), reverse=True)  # sort rows by gold
        with open(os.path.join(experiment_dir, output_filename + '.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'bws', 'predicted', 'sentence'])
            writer.writerows(rows)

    pearson = pearsonr(gold_v, gppl_v)
    spearman = spearmanr(gold_v, gppl_v)
    print('Pearson:', pearson)
    print('Spearman:', spearman)
    print('')

    # create plot
    plt.ioff()
    plt.scatter(gold_v, gppl_v)
    plt.savefig(os.path.join(experiment_dir, output_filename + '.png'))
    plt.close()

    # append metadata to overview file
    OPTIONS['1_id'] = output_filename
    OPTIONS['2_count'] = len(dev_idxs)
    OPTIONS['spearman'] = spearman
    OPTIONS['pearson'] = pearson
#    OPTIONS['runtime'] = '{:.2f}'.format((datetime.now() - start_time).total_seconds() / 60)
    OPTIONS['runtime'] = str(datetime.now() - start_time).split('.')[0]

    overview_path = os.path.join(experiment_dir, task + '_' + overview_file)
    if not os.path.exists(overview_path):
        with open(overview_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(sorted(OPTIONS.keys()))
    with open(overview_path, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(OPTIONS.keys()))
        writer.writerow(OPTIONS)



param_grid = OrderedDict()
#param_grid['emb_path'] = ['/home/dodinh/data/embeddings/word2vec/GoogleNews-vectors-negative300.vocab_sub_']
param_grid['optimization'] = [False]  #, True]
param_grid['add_features'] = [[]]
#param_grid['pkl_data_path'] = ['{}_{}_data.pkl']
param_grid['data_mode'] = ['full']


# run grid of options
if __name__ == '__main__':

    if len(sys.argv) < 1:
        print('Usage: run_task1_experiment [humour|metaphor]')
        exit()

    task = sys.argv[1]
    experiment_dir = './results/experiment_{}_{}'.format(task, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(experiment_dir)

    if task == 'metaphor':
        param_grid['data_mode'] = ['pairs']

    grid = ParameterGrid(param_grid)
    logging.info('Starting [%d] experiments run to "%s"...', len(grid), experiment_dir)

    for OPTIONS in grid:
        random.seed(rnd_seed)
        np.random.seed(np_seed)
        run(task, experiment_dir)
