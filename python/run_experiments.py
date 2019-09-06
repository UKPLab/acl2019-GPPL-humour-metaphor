'''
Simpler example showing how to use train and use the convincingness model for prediction.

This script trains a model on the UKPConvArgStrict dataset. So, before running this script, you need to run
"python/analysis/habernal_comparison/run_preprocessing.py" to extract the linguistic features from this dataset.

'''
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')

import sys
# include the paths for the other directories
sys.path.append("./python")
sys.path.append("./python/analysis")
sys.path.append("./python/models")
sys.path.append("./python/analysis/habernal_comparison")

from gp_pref_learning import GPPrefLearning

from collections import defaultdict, namedtuple, OrderedDict
from datetime import datetime
from itertools import chain, combinations
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from operator import itemgetter
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import ParameterGrid
from vuamc import Vuamc

import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pickle
import random
import re


# data paths
HUMOUR_DATA_PATH = './data/pl-humor-full/results.tsv'
HUMOUR_BWS_SCORES_PATH = './data/bws/item_scores.txt'
METAPHOR_DATA_PATH = './data/vuamc_crowd/all.csv'
VUAMC_PATH = './data/VU Amsterdam Metaphor Corpus/2541/VUAMC_with_novelty_scores.xml'
NGRAMS_PATH = './data/ngrams/en/{}_{}grams.csv'  # task, n
FREQUENCIES_PATH = './data/resources/wikipedia_2017_frequencies.pkl'

# set numpy seed
rnd_seed = 41
np_seed = 1337

# ngram settings
n = 2


def load_embeddings(path):
    """
    Load embeddings from file and put into dict.

    :param path: path to embeddings file
    :return: a map word->embedding
    """
    logging.info('Loading embeddings...')
    embeddings = dict()
    with open(path, 'r') as f:
        for line in f:
            line = line.split(' ')
            embeddings[line[0]] = np.array([float(a) for a in line[1:]])
    return embeddings


def split_data(pairs, idx_instance_list, train_size=0.6, dev_size=0.2, test_size=0.2, cut=None, cut_mode=None):
    """
    Shuffle and split data into train, dev, test sets.

    :param pairs: list of available data samples (index-pairs)
    :idx_instance_list: list of instances, for which the position represents the index
    :param train_size: ratio of data to be used as training set
    :param dev_size: ratio of data to be used as dev set
    :param test_size: ratio of data to be used as test set
    :param cut: ratio of training data that should actually be used
    :param cut_mode: 'random_pairs' or 'random_instances'
    """
    assert train_size + dev_size + test_size == 1
    logging.info('Splitting data (%s/%s/%s)...', train_size, dev_size, test_size)

    idxs = list(range(len(idx_instance_list)))
    random.shuffle(idxs)
    logging.info('Shuffling data... [%s, ...]', ', '.join([str(i) for i in idxs[:10]]))

    train_idxs = idxs[:int(len(idxs) * train_size)]
    dev_idxs = idxs[int(len(idxs) * train_size):-int(len(idxs) * test_size)]
    test_idxs = idxs[int(-len(idxs) * test_size):]

    # only use pairs for training which have both idxs in the train_idxs set
    train_pairs = []
    for pair in pairs:
        if pair[0] in train_idxs and pair[1] in train_idxs:
            train_pairs.append(pair)

    # cut training set size; do this after removing instances from dev/test,
    # because this way dev/test are not being changed
    if cut is not None:
        cut = float(cut)
        assert 0 <= cut <= 1
        logging.info('Using only a portion of the training data: %s', cut)
        # we take randomly selected pairs according to the cut
        if cut_mode == 'random_instances':
            random.shuffle(train_pairs)
            cut_train_pairs = train_pairs[:int(cut*len(train_pairs))]
        # we take all instances of randomly selected pairs according to the cut
        elif cut_mode == 'random_pairs':
            def get_pid(pair):
                return '#'.join([str(i) for i in sorted(pair)])
            pair_ids = list(set([get_pid(pair) for pair in train_pairs]))
            random.shuffle(pair_ids)
            pair_ids = pair_ids[:int(cut*len(pair_ids))]
            cut_train_pairs = [p for p in train_pairs if get_pid(p) in pair_ids]
        else:
            raise ValueError('cut_mode needs to be one of [random_instances, random_pairs]')
        logging.info('Using "%s", choosing %s train_pairs...', cut_mode, len(cut_train_pairs))

        # change training idxs as well, to enable faster feature extraction
        remaining_idxs = set()
        for pair in train_pairs:
            remaining_idxs.add(pair[0])
            remaining_idxs.add(pair[1])
        train_idxs = [i for i in train_idxs if i in remaining_idxs]

        # write out used pairs for training, dev, test; to create "reduced" BWS scores
        with open(os.path.join(OPTIONS['experiment_dir'], 'items_{}_{}.txt'.format(str(cut), cut_mode)), 'w') as f:
            writer = csv.DictWriter(f, fieldnames=['Item1','Item2','Best','Worst'])
            writer.writeheader()
            for pair in cut_train_pairs:
                row = {'Item1': pair[0], 'Item2': pair[1], 'Best': pair[0], 'Worst': pair[1]}
                writer.writerow(row)
        with open(os.path.join(OPTIONS['experiment_dir'], 'items_instances_{}_{}.txt'.format(str(cut), cut_mode)), 'w') as f:
            for idx, instance in enumerate(idx_instance_list):
                if isinstance(instance, list):
                    f.write('{}\t{}\n'.format(idx, ' '.join(instance)))
                elif isinstance(instance, str):
                    f.write('{}\t{}\n'.format(idx, instance))
                else:
                    raise ValueError('Instance should be list (humour) or str (metaphor).')
        train_pairs = cut_train_pairs

    return train_pairs, train_idxs, dev_idxs, test_idxs


def get_instance(instance_id):
    """
    Get the textual representation corresponding to a given instance_id, depending on the task.
    For metaphor, instance_id is an index id to retrieve data from the VUAMC.
    For humour, instance_id is a sentence.
    """
    if OPTIONS['task'] == 'metaphor':
        _, sentence, metaphor = VUAMC.get(instance_id)
        focus_position = int(instance_id.split('.')[2])  # 0-based index of the metaphor token in the sentence
        return [t.covered_text.lower() for t in sentence.tokens], focus_position
    elif OPTIONS['task'] == 'humour':
        return [t.lower() for t in instance_id], None


def mean_wikipedia_frequency(tokens):
    """
    Retrieves frequencies for a list of tokens and returns mean frequency.
    """
    freq_sum = 0.0
    for token in tokens:
        lemma = LEMMATIZER.lemmatize(token)
        freq_sum = FREQUENCY_CACHE.get(lemma, 1)
    return freq_sum / len(tokens)


def mean_wordnet_polysemy(tokens):
    """
    Retrieves polysemy values (WordNet synset counts) for a list of tokens and returns mean polysemy.
    """
    synset_count = 0.0
    for token in tokens:
        lemma = LEMMATIZER.lemmatize(token)
        synset_count += len(wn.synsets(lemma))
    return synset_count / len(tokens)


def mean_ngram_frequency(tokens):
    """
    Calculates mean ngram frequency for a list of tokens based on Google ngrams.
    """
    if len(tokens) < n:
        return 0
    ngrams = list(nltk.ngrams(tokens, n))
    total = sum([NGRAM_CACHE[' '.join(ngram)] for ngram in ngrams])
    return total / len(ngrams)


def load_crowd_data(task):
    if task == 'metaphor':
        return load_metaphor_data(METAPHOR_DATA_PATH)
    elif task == 'humour':
        return load_humour_data(HUMOUR_DATA_PATH)
    else:
        raise ValueError('task must be in [metaphor, humour]')


def load_humour_data(path):
    """
    Read humour csv and create preference pairs of tokenized sentences.
    The csv format is: id,selection,instance_A,instance_B

    :param path: path to crowdsourced data
    :return: a list of index pairs, a list of instances
    """
    logging.info('Loading humour crowd data from %s...', path)
    pairs = []
    idx_instance_list = []
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # skip header row
        for line_no, line in enumerate(reader):
            answer = line[1]
            A = word_tokenize(line[2])
            B = word_tokenize(line[3])
            # add instances to list (if not already in it)
            if A not in idx_instance_list:
                idx_instance_list.append(A)
            if B not in idx_instance_list:
                idx_instance_list.append(B)
            # add pair to list if there is a preference (in decreasing preference order)
            if answer == 'A':
                pairs.append((idx_instance_list.index(A), idx_instance_list.index(B)))
            if answer == 'B':
                pairs.append((idx_instance_list.index(B), idx_instance_list.index(A)))
    return pairs, idx_instance_list


def load_metaphor_data(path):
    """
    Read csv and create preference pairs of VUAMC ids representing sentences with focus.
    The csv format: 3 BWS-comparisons per line

    :param path: path to crowdsourced data
    :return: a list of index pairs, a list of instances
    """
    logging.info('Loading metaphor crowd data from %s...', path)
    pairs = []
    idx_instance_list = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for line_no, line in enumerate(reader):
            for i in range(0,3):  # 3 comparisons per HIT/line
                # add instances to list (if not alreay in it)
                bws_vuamc_ids = OrderedDict(zip(line[27+(i*16):27+(i*16) + 4], line[27+(i*16)+4:27+(i*16) + 4 + 4]))
                con_id = bws_vuamc_ids[line[87+(i*2)]] if line[87+(i*2)] in bws_vuamc_ids else None  # most conventionalized (i.e. "worst")
                nov_id = bws_vuamc_ids[line[88+(i*2)]] if line[88+(i*2)] in bws_vuamc_ids else None  # most novel (i.e. "best")
                # check hit validity, otherwise skip
                if nov_id is None or con_id is None or None in [re.match(r'^\d+\.\d+\.\d+$', vuamc_id) for vuamc_id in bws_vuamc_ids.values()]:
                    continue
                for vuamc_id in bws_vuamc_ids.values():
                    if vuamc_id not in idx_instance_list:
                        idx_instance_list.append(vuamc_id)
                # in "pairs" mode we just use the "best vs worst" comparison
                if OPTIONS['data_mode'] == 'pairs':
                    pairs.append((idx_instance_list.index(nov_id), idx_instance_list.index(con_id)))
                # otherwise we use all induced pairs, e.g. the "best" instance is better than all others, ...
                else:
                    for current_id in bws_vuamc_ids.values():
                        if current_id != nov_id:  # best (a) > b, c, d
                            pairs.append((idx_instance_list.index(nov_id), idx_instance_list.index(current_id)))  # append "novel" pairs
                            if current_id != con_id:  # worst (d) < b, c
                                pairs.append((idx_instance_list.index(current_id), idx_instance_list.index(con_id)))  # append "conventionalized" pairs
    return pairs, idx_instance_list


def extract_features(requested_idxs, idx_instance_list, embeddings, add_feats):
    """
    Extract features for the metaphors in the given list.

    :param idx_pairs: a list of preference pairs of indexes
    :param idx_instance_list: a list of indexed instances
    :param embeddings: a map token->embedding
    :param add_feats: if True, additional features are added to the vectors
    :return: numpy array of feature vectors  # , list of connected vuamc_ids
    """
    logging.info('Extracting features for %s instances...', len(requested_idxs))

    metaphor_mode = 'concat'  # 'double'
    if OPTIONS['task'] == 'metaphor':
        logging.info('Metaphor feature vector mode: %s', metaphor_mode)

#    emb_dim = embeddings[list(embeddings.keys())[0]].shape[0]
    emb_dim = 300
    empty = np.zeros(emb_dim)
    feature_vectors = []

    # extract features for requested indexes
    for idx in requested_idxs:
        try:
            tokens, focus_position = get_instance(idx_instance_list[idx])
        except:
            logging.error('VUAMC id not found, returning zero-vector: [%s] [%s]', idx, idx_instance_list[idx])
            return np.concatenate([empty, np.zeros(len(add_feats) + (1 if OPTIONS['task'] == 'metaphor' else 0))])
        tokens = [t.lower() for t in tokens]  # lowercase tokens
        embs = [embeddings.get(t) for t in tokens if t in embeddings]
        # no embeddings found: we use a zero embedding
        if len(embs) == 0:
            embs = [empty]
        # metaphor: we use two different variants for feature representation
        if OPTIONS['task'] == 'metaphor':
            focus_emb = embeddings.get(tokens[focus_position], empty)
            if metaphor_mode == 'double':  # for this mode, we double the occurrence of the metaphor token
                embs.append(focus_emb)
            fv = np.mean(embs, axis=0)
            if metaphor_mode == 'concat':  # for this mode, we concatenate the metaphor embedding to the mean embedding
                fv = np.concatenate([fv, focus_emb])
        # humour: use mean embedding as main feature vector
        elif OPTIONS['task'] == 'humour':
            fv = np.mean(embs, axis=0)

        # feature: mean frequency
        if 'frequency' in add_feats:
            mean_frequency = np.array([mean_wikipedia_frequency(tokens)])
            fv = np.concatenate([fv, mean_frequency])
            if OPTIONS['task'] == 'metaphor':
                focus_frequency = np.array([FREQUENCY_CACHE.get(tokens[focus_position], 1)])
#                focus_frequency = np.array([mean_wikipedia_frequency(tokens[focus_position:focus_position+1])])
                fv = np.concatenate([fv, focus_frequency])
        # feature: mean polysemy
        if 'polysemy' in add_feats:
            mean_polysemy = np.array([mean_wordnet_polysemy(tokens)])
            fv = np.concatenate([fv, mean_polysemy])
        # feature: mean ngram frequency
        if 'ngrams' in add_feats:
            mean_ngram_freq = np.array([mean_ngram_frequency(tokens)])
            fv = np.concatenate([fv, mean_ngram_freq])

        feature_vectors.append(fv)

    logging.info('FVs %s, IDXs %s, IDX_DATA_MAP %s', len(feature_vectors), len(requested_idxs), len(idx_instance_list))

    feature_vectors = np.array(feature_vectors)
    # we normalize the extra feature dimensions (each on their own), because otherwise they throw off the model
    emb_dim = 2 * emb_dim if OPTIONS['task'] == 'metaphor' and metaphor_mode == 'concat' else emb_dim  # size of the concatenated embeddings without additional features
    for dim in range(emb_dim, feature_vectors.shape[1]):
        norm_factor = max(abs(np.max(feature_vectors[:,dim])), abs(np.min(feature_vectors[:,dim])))
        feature_vectors[:,dim] = feature_vectors[:,dim] / norm_factor
        logging.info('  Normalized dimension %s by factor %s', dim, norm_factor)
    return feature_vectors


def train(training_split, train_idxs, idx_instance_list, embeddings):
    """
    Train a model using the given training set.

    :param training_split: list of training examples (index-pairs)
    :param idx_vuamc_map: 
    """
    logging.info('Training model...')

    # needs to be mapped to the ids used in the training_splits
    items_feat = extract_features(train_idxs, idx_instance_list, embeddings, add_feats=OPTIONS['add_features'])  
    a1_train, a2_train = zip(*training_split)

    # re-assign indexes based on position in the index list (this makes a*_train compatible with the order of items_feat)
    a1_train = [train_idxs.index(idx) for idx in a1_train]
    a2_train = [train_idxs.index(idx) for idx in a2_train]

#    ls_initial = compute_median_lengthscales(items_feat)
#    rate_ls=1.0 / np.mean(ls_initial),
    ls_initial = None  # default
    rate_ls = 10  # default

    # following ES' advice here
    ninducing = min(len(train_idxs), OPTIONS['ninducing'])

    model = GPPrefLearning(ninput_features=items_feat.shape[1],
                           kernel_func=OPTIONS['kernelfunc'],  # 'matern_3_2' is default, 'diagonal' is used for testing without features
                           ls_initial=ls_initial,
                           verbose=False,
                           shape_s0=2.0,
                           rate_s0=200.0,
                           rate_ls=rate_ls,
                           use_svi=True,
                           ninducing=ninducing,
                           max_update_size=200,
                           kernel_combination='*',
                           forgetting_rate=0.7,
                           delay=1.0)

    model.max_iter_VB = 2000

    # in the data loading step we already sorted each pair, preference-descending wise,
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


def run(cut, cut_mode):
    """
    Run a given configuration.
    """
    logging.info('Running GPPL on task [%s] using these options: %s', OPTIONS['task'], OPTIONS)
    output_filename = 'results-' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    start_time = datetime.now()

    # data and embeddings loading
    # TODO we can cut loading times by doing this only once, instead of for each run
    # currently, we retain the flexibility of loading e.g. different embeddings
    pairs, idx_instance_list = load_crowd_data(OPTIONS['task'])
    embeddings = load_embeddings(path=OPTIONS['emb_path'] + OPTIONS['task'])

    # training/dev/test set randomization and splitting
    if OPTIONS['task4']:
        train_perc = 1
        dev_perc = 0
        test_perc = 0
        train_split, train_idxs, dev_idxs, test_idxs = split_data(pairs, idx_instance_list, train_perc, dev_perc, test_perc, cut=cut, cut_mode=cut_mode)
        logging.warn('Training split is 100%, using complete set as test set (emulating data that is available to BWS)...')
        dev_idxs = train_idxs
    else:
        train_perc = 0.6
        dev_perc = 0.2
        test_perc = 0.2
        train_split, train_idxs, dev_idxs, test_idxs = split_data(pairs, idx_instance_list, train_perc, dev_perc, test_perc, cut=cut, cut_mode=cut_mode)

    # train model on cut
    model = train(train_split, train_idxs, idx_instance_list, embeddings)
    save_model(model, OPTIONS['experiment_dir'], output_filename)

    # apply model to dev split
    dev_feats = extract_features(dev_idxs, idx_instance_list, embeddings, add_feats=OPTIONS['add_features'])
    predicted_f, _ = model.predict_f(out_feats=dev_feats)
    predicted = dict(zip(dev_idxs, [float(p) for p in predicted_f]))

    rows = []
    gold_v = []
    gppl_v = []

    if OPTIONS['task'] == 'metaphor':
        for dev_idx, pred in predicted.items():
            vuamc_id = idx_instance_list[dev_idx]
            _, sentence, metaphor = VUAMC.get(vuamc_id)
            rows.append([vuamc_id, metaphor.score, pred, metaphor.covered_text, sentence.pp()])
            if metaphor.score is not None and pred is not None:
                gold_v.append(float(metaphor.score))
                gppl_v.append(pred)
        fieldnames = ['id', 'novelty', 'predicted', 'metaphor', 'sentence']
    elif OPTIONS['task'] == 'humour':
        scores = dict()
        with open(HUMOUR_BWS_SCORES_PATH, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                scores[' '.join(word_tokenize(line[2]))] = float(line[1])
        for dev_idx, pred in predicted.items():
            instance = idx_instance_list[dev_idx]
            bws = scores[' '.join(instance)]  # get gold score
            gold_v.append(bws)
            gppl_v.append(pred)
            rows.append([dev_idx, bws, pred, ' '.join(instance)])
        fieldnames = ['id', 'bws', 'predicted', 'sentence']

    # write output
    rows = sorted(rows, key=itemgetter(1), reverse=True)  # sort rows by gold
    with open(os.path.join(OPTIONS['experiment_dir'], output_filename + '.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        writer.writerows(rows)

    pearson = pearsonr(gold_v, gppl_v)
    spearman = spearmanr(gold_v, gppl_v)
    print('Pearson:', pearson)
    print('Spearman:', spearman)
    print('')

    # create plot
    plt.ioff()
    plt.scatter(gold_v, gppl_v)
    plt.savefig(os.path.join(OPTIONS['experiment_dir'], output_filename + '.png'))
    plt.close()

    # append metadata to overview file
    OPTIONS['1_id'] = output_filename
    OPTIONS['2_count'] = len(dev_idxs)
    OPTIONS['spearman'] = spearman
    OPTIONS['pearson'] = pearson
    OPTIONS['runtime'] = str(datetime.now() - start_time).split('.')[0]
    OPTIONS['cut'] = str(cut)
    OPTIONS['cut_mode'] = cut_mode
    overview_path = os.path.join(OPTIONS['experiment_dir'], OPTIONS['task'] + '_overview.csv')
    if not os.path.exists(overview_path):
        with open(overview_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(sorted(OPTIONS.keys()))
    with open(overview_path, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(OPTIONS.keys()))
        writer.writerow(OPTIONS)


param_grid = OrderedDict()
param_grid['kernelfunc'] = ['matern_3_2']  # ['diagonal']
param_grid['ninducing'] = [500]
param_grid['emb_path'] = ['data/GoogleNews-vectors-negative300.vocab_sub_']
param_grid['optimization'] = [False]  #, True]
# no features, all, and ablation
features = ['frequency', 'polysemy', 'ngrams']  # available features
param_grid['add_features'] = list(chain.from_iterable(combinations(features, r) for r in [0, len(features) - 1, len(features)]))
param_grid['task4'] = [False]

# run grid of options
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Usage: run_experiments.py task cuts(comma-separated list of percentages of training data to use)')
        exit()

    task = sys.argv[1]
    param_grid['cut'] = [float(c) for c in sys.argv[2].split(',')] if len(sys.argv) > 2 else [None]
    if len(sys.argv) > 3 and sys.argv[3] == 'task4':
        param_grid['task4'] = [True]
    experiment_dir = 'experiment_{}_{}'.format(task, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(experiment_dir)

    if task == 'metaphor':
        VUAMC = Vuamc(VUAMC_PATH)
        # only run on "original BWS" pairs for now
        param_grid['data_mode'] = ['pairs']  # ['full']
#    elif task == 'humour':
#        pass
#    else:
#        raise ValueError('task must be in [metaphor, humour]')

    # use only the best performing feature selection for cuts
    if param_grid['cut'] == [None]:
        param_grid['cut_mode'] = [None]
    else:
        param_grid['cut_mode'] = ['random_instances', 'random_pairs']
        if task == 'metaphor':
            param_grid['add_features'] = [('frequency', 'ngrams')]
            logging.warn('Using only best feature combination: %s', param_grid['add_features'])
        elif task == 'humour':
            param_grid['add_features'] = [('frequency', 'ngrams')]
            logging.warn('Using only best feature combination: %s', param_grid['add_features'])

    # initialize tools and resources
    logging.info('Initializing lemmatizer...')
    LEMMATIZER = WordNetLemmatizer()
    logging.info('Loading frequencies from pkl file...')
    with open(FREQUENCIES_PATH, 'rb') as f:
        FREQUENCY_CACHE = pickle.load(f)
        logging.debug('freq has %s entries.', len(FREQUENCY_CACHE))
    logging.info('Initializing ngram cache...')
    with open(NGRAMS_PATH.format(task, n), 'r') as f:
        NGRAM_CACHE = defaultdict(int)
        for line in f:
            line = line.strip().split('\t')
            NGRAM_CACHE[line[0]] = int(line[1])

    param_grid['task'] = [task]
    param_grid['experiment_dir'] = [experiment_dir]
    grid = ParameterGrid(param_grid)
    logging.info('Starting [%d] experiments run to "%s"...', len(grid), experiment_dir)

    for OPTIONS in grid:
        # set seeds so they are the same for each feature run
        random.seed(rnd_seed)
        np.random.seed(np_seed)

        run(OPTIONS['cut'], OPTIONS['cut_mode'])
