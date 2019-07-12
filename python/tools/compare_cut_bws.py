'''
Simpler example showing how to use train and use the convincingness model for prediction.

This script trains a model on the UKPConvArgStrict dataset. So, before running this script, you need to run
"python/analysis/habernal_comparison/run_preprocessing.py" to extract the linguistic features from this dataset.

'''
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s')
#console = logging.StreamHandler()
#console.setFormatter(logging.Formatter('%(asctime)s %(levelname)s\t%(message)s'))
#logging.getLogger('').handlers = []
#logging.getLogger('').addHandler(console)
#logging.getLogger('gp_pref_learning').setLevel(logging.DEBUG)
#logging.getLogger('matplotlib').setLevel(logging.WARNING)


import sys
# include the paths for the other directories
sys.path.append("./python")
sys.path.append("./python/analysis")
sys.path.append("./python/models")
sys.path.append("./python/analysis/habernal_comparison")


from gp_classifier_vb import compute_median_lengthscales
from gp_pref_learning import GPPrefLearning
from tests import get_docidxs_from_ids, get_doc_token_seqs, get_mean_embeddings

#from bidict import bidict
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
import emb_utils
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ngram_utils
import ngram_utils2
import nltk
import numpy as np
import os
import pickle
import random
import re


# data paths
#humour_path = '/home/dodinh/humour.tsv'
humour_path = '/home/dodinh/pl-humor-full/results.tsv'
#metaphor_path = 'python/data/vuamc_crowd/1/2nd_round_211_to_2210.csv'
metaphor_path = 'python/data/vuamc_crowd/all.csv'
vuamc_path = '/home/dodinh/data/VU Amsterdam Metaphor Corpus/2541/VUAMC_with_novelty_scores.xml'
overview_file = 'overview.csv'

# set numpy seed
rnd_seed = 41
np_seed = 1337

# ngram settings
n = 2


#
# independent of data -------------------------------------------------------------------------------
#
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


def split_data(pairs, idx_instance_list, train_size=0.6, dev_size=0.2, test_size=0.2, seed=None, cut=None, cut_mode=None, experiment_dir=None):
    """
    Shuffle and split data into train, dev, test sets.

    :param data: list of available data samples (index-pairs)
    :param train_size: ratio of data to be used as training set
    :param dev_size: ratio of data to be used as dev set
    :param test_size: ratio of data to be used as test set
    :param seed: if set, randomizes the input data before splitting
    :param cut: ratio of training data that should actually be used
    :param cut_mode: 'random_pairs' or 'random_instances'
    """
    assert train_size + dev_size + test_size == 1
    logging.info('Splitting data (%s/%s/%s)...', train_size, dev_size, test_size)

    idxs = list(range(len(idx_instance_list)))
    if seed is not None:
#        random.seed(rnd_seed)  # we set the seed before running each experiment
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
        logging.info('Using "%s", choosing %s train_pairs...', cut_mode, len(train_pairs))

        # change training idxs as well, to enable faster feature extraction
        remaining_idxs = set()
        for pair in train_pairs:
            remaining_idxs.add(pair[0])
            remaining_idxs.add(pair[1])
        train_idxs = [i for i in train_idxs if i in remaining_idxs]

        # write out used pairs for training, dev, test; to create "reduced" BWS scores
        with open(os.path.join(experiment_dir, 'items_{}_cut.txt'.format(str(cut))), 'w') as f:
            writer = csv.DictWriter(f, fieldnames=['Item1','Item2','Best','Worst'])
            writer.writeheader()
            reduced_pairs = [p for p in pairs if p not in train_pairs] + cut_train_pairs
            for pair in reduced_pairs:  # i.e.: pairs - (train_pairs - cut_train_pairs)
                row = {'Item1': pair[0], 'Item2': pair[1], 'Best': pair[0], 'Worst': pair[1]}
                writer.writerow(row)
        with open(os.path.join(experiment_dir, 'items_instances_{}_cut.txt'.format(str(cut))), 'w') as f:
            for idx, instance in enumerate(idx_instance_list):
                if instance is list:
                    f.write('{}\t{}\n'.format(idx, ' '.join(instance)))
                elif instance is str:
                    f.write('{}\t{}\n'.format(idx, instance))
                else:
                    raise ValueError('Instance should be list (humour) or str (metaphor).')

        train_pairs = cut_train_pairs

    return train_pairs, train_idxs, dev_idxs, test_idxs


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


def mean_wikipedia_frequency(frequency_cache, lemmatizer, tokens):
    """
    Retrieves frequency for a list of tokens and returns mean frequency.

    :param frequency_cache: a frequencey lookup table
    :param lemmatizer: a lemmatizer
    :param tokens: a sequence of tokens (strings)
    """
    freq_sum = 0
    for token in tokens:
        lemma = lemmatizer.lemmatize(token)
        freq_sum = frequency_cache.get(lemma, 1)
    return freq_sum / len(tokens)


def mean_wordnet_polysemy(lemmatizer, tokens):
    """
    Retrieves polysemy for a list of tokens and returns mean polysemy.

    :param lemmatizer: a lemmatizer
    :param tokens: a sequence of tokens (strings)
    """
    synset_count = 0
    for token in tokens:
        lemma = lemmatizer.lemmatize(token)
        synset_count += len(wn.synsets(lemma))
    return synset_count / len(tokens)


def mean_ngram_frequency(ngram_cache, tokens):
    """
    Calculates mean ngram frequency for a list of tokens based on Google ngrams.
    """
    if len(tokens) < n:
        return 0
    ngrams = list(nltk.ngrams(tokens, n))
    total = sum([ngram_cache[' '.join(ngram)] for ngram in ngrams])
    return total / len(ngrams)



#
# dependent of concrete data -------------------------------------------------------------------------------
#
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


def extract_features(requested_idxs, idx_instance_list, embeddings, vuamc, add_feats, task):
    """
    Extract features for the metaphors in the given list.

    :param idx_pairs: a list of preference pairs of indexes
    :param idx_instance_list: a map idx->vuamc-id
    :param embeddings: a map token->embedding
    :param vuamc: a Vuamc object containing the structured VUAMC
    :param add_feats: if True, additional features are added to the vectors
    :return: numpy array of feature vectors  # , list of connected vuamc_ids
    """
    logging.info('Extracting features...')

    metaphor_mode = 'concat'  # 'double'
    if task == 'metaphor':
        logging.info('Metaphor feature vector mode: %s', metaphor_mode)

#    emb_dim = embeddings[list(embeddings.keys())[0]].shape[0]
    emb_dim = 300
    empty = np.zeros(emb_dim)
    feature_vectors = []

    # load caches and initialize tools
    if 'frequency' in add_feats:
        logging.info('Loading frequencies from pkl file...')
        with open('freqs.pkl', 'rb') as f:
            frequency_cache = pickle.load(f)
            logging.debug('freq has %s entries.', len(frequency_cache))
        logging.info('Initializing lemmatizer...')
        lemmatizer = WordNetLemmatizer()
    elif 'polysemy' in add_feats:
        logging.info('Initializing lemmatizer...')
        lemmatizer = WordNetLemmatizer()
    if 'ngrams' in add_feats:
        logging.info('Initializing ngram cache...')
        ngram_cache = defaultdict(int)
        with open('/home/dodinh/data/ngrams/{}_{}grams.csv'.format(task, n), 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                ngram_cache[line[0]] = int(line[1])

    # extract features for requested indexes
    for idx in requested_idxs:
        try:
            tokens, focus_position = get_instance(task, idx_instance_list[idx], vuamc)
        except:
            logging.error('VUAMC id not found, returning zero-vector: [%s] [%s]', idx, idx_instance_list[idx])
            return np.concatenate([empty, np.zeros(len(add_feats) + (1 if task == 'metaphor' else 0))])
        tokens = [t.lower() for t in tokens]  # lowercase tokens
        embs = [embeddings.get(t) for t in tokens if t in embeddings]
        # no embeddings found: we use a zero embedding
        if len(embs) == 0:
            embs = [empty]
        # metaphor: we use two different variants for feature representation
        if task == 'metaphor':
            focus_emb = embeddings.get(tokens[focus_position], empty)
            if metaphor_mode == 'double':  # for this mode, we double the occurrence of the metaphor token
                embs.append(focus_emb)
            fv = np.mean(embs, axis=0)
            if metaphor_mode == 'concat':  # for this mode, we concatenate the metaphor embedding to the mean embedding
                fv = np.concatenate([fv, focus_emb])
        # humour: use mean embedding as main feature vector
        elif task == 'humour':
            fv = np.mean(embs, axis=0)

        # feature: mean frequency
        if 'frequency' in add_feats:
            mean_frequency = np.array([mean_wikipedia_frequency(frequency_cache, lemmatizer, tokens)])
            fv = np.concatenate([fv, mean_frequency])
            if task == 'metaphor':
                focus_frequency = np.array([frequency_cache.get(tokens[focus_position], 1)])
                fv = np.concatenate([fv, focus_frequency])
        # feature: mean polysemy
        if 'polysemy' in add_feats:
            mean_polysemy = np.array([mean_wordnet_polysemy(lemmatizer, tokens)])
            fv = np.concatenate([fv, mean_polysemy])
        # feature: mean ngram frequency
        if 'ngrams' in add_feats:
            mean_ngram_freq = np.array([mean_ngram_frequency(ngram_cache, tokens)])
            fv = np.concatenate([fv, mean_ngram_freq])

        feature_vectors.append(fv)

#    logging.debug("Instances without embedding:", len(no_embed))
    logging.info('FVs %s, IDXs %s, IDX_DATA_MAP %s', len(feature_vectors), len(requested_idxs), len(idx_instance_list))

    feature_vectors = np.array(feature_vectors)
    print('shape', feature_vectors.shape)
    # we normalize the extra feature dimensions (each on their own), because otherwise they throw off the model
    emb_dim = 2 * emb_dim if task == 'metaphor' and metaphor_mode == 'concat' else emb_dim  # size of the concatenated embeddings without additional features
    for dim in range(emb_dim, feature_vectors.shape[1]):
        norm_factor = max(abs(np.max(feature_vectors[:,dim])), abs(np.min(feature_vectors[:,dim])))
        feature_vectors[:,dim] = feature_vectors[:,dim] / norm_factor
        logging.info('  Normalized dimension %s by factor %s', dim, norm_factor)
    return feature_vectors


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


def run(task, cut, cut_mode, experiment_dir):
    """
    Run a given configuration.
    """
    logging.info('Running GPPL on task [%s] using these options: %s', task, OPTIONS)
    output_filename = 'results-' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    start_time = datetime.now()

    # data loading
    pkl_data_path = OPTIONS['pkl_data_path'].format(task, OPTIONS['data_mode'])
    if os.path.exists(pkl_data_path):  # load data form pkl (faster)
        if task == 'humour':
            vuamc = None
        elif task == 'metaphor':
            vuamc = Vuamc(vuamc_path)
        logging.info('Loading data from pkl: %s', pkl_data_path)
        with open(pkl_data_path, 'rb') as f:
            pairs, idx_instance_list = pickle.load(f)
    else:
        if task == 'humour':
            pairs, idx_instance_list = load_crowd_data_TM(humour_path)
            vuamc = None
        elif task == 'metaphor':
            pairs, idx_instance_list = load_crowd_data_ED(metaphor_path)
            vuamc = Vuamc(vuamc_path)
        logging.info('Saving data to pkl: %s', pkl_data_path)
        with open(pkl_data_path, 'wb') as f:
            pickle.dump((pairs, idx_instance_list), f)

    # training/dev/test set randomization and splitting
    embeddings = load_embeddings(path=OPTIONS['emb_path'] + task)
    train_perc = 0.6
    dev_perc = 0.2
    test_perc = 0.2
    train_split, train_idxs, dev_idxs, test_idxs = split_data(pairs, idx_instance_list, train_perc, dev_perc, test_perc, seed=rnd_seed, cut=cut, cut_mode=cut_mode, experiment_dir=experiment_dir)

    # train model on cut
    model = load_model(TODO)

    # apply model to dev split
    dev_feats = extract_features(dev_idxs, idx_instance_list, embeddings, vuamc, add_feats=OPTIONS['add_features'], task=task)
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
                print("V:", vuamc_ids[i])
                print("N:", metaphor.score)
                print("P:", predicted_f[i][0])
                raise e

        # write output
        rows = sorted(rows, key=itemgetter(1), reverse=True)  # sort rows by gold
        with open(os.path.join(experiment_dir, output_filename + '.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'novelty', 'predicted', 'metaphor', 'sentence'])
            writer.writerows(rows)
    else:
        scores = dict()
        with open('bws/item_scores.txt', 'r') as f:
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
    OPTIONS['runtime'] = str(datetime.now() - start_time).split('.')[0]
    OPTIONS['cut'] = str(cut)
    OPTIONS['cut_mode'] = cut_mode
    overview_path = os.path.join(experiment_dir, task + '_' + overview_file)
    if not os.path.exists(overview_path):
        with open(overview_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(sorted(OPTIONS.keys()))
    with open(overview_path, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(OPTIONS.keys()))
        writer.writerow(OPTIONS)


def output(experiment_file):
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
        plt.savefig(experiment_file + '_cats.png'))
        plt.close()
    else:
        raise ValueError('task needs to be one of [metaphor, humor]')

id, bws, category
id,bws,predicted,category,sentence
2441,0.709,0.8344922877273088,hompun,I do a lot of spreadsheets in the office so you can say I 'm excelling at work .

id,novelty,predicted,metaphor,sentence
20.26.50,0.765,0.2895287893868407,goose-step,"Ron Todd, the general secretary of the transport workers' union, said that today's vote on the multilaterist nuclear defence policy would not yield the ‘massive and overwhelming’ majority predicted by wingersright-, and he warned that party leaders could not expect everybody to ‘goose-step ’ in the same direction once the policy had been carried."
95.303.26,0.733,-0.0027180424546173493,monkeys,"so he's killing himself laughing I I've got to tell you Ron, I've been taping he said I don't give a monkeys you tape what, he said let them all hear, they'll all be able to pass it round!"


param_grid = OrderedDict()
#param_grid['emb_path'] = ['/home/dodinh/data/embeddings/word2vec/GoogleNews-vectors-negative300.vocab_sub_',
#                          '/home/dodinh/data/embeddings/komninos/wiki_extvec_']
param_grid['emb_path'] = ['/home/dodinh/data/embeddings/word2vec/GoogleNews-vectors-negative300.vocab_sub_']
param_grid['optimization'] = [False]  #, True]

features = ['frequency', 'polysemy', 'ngrams']  # available features
# all feature combinations
#param_grid['add_features'] = list(chain.from_iterable(combinations(features, r) for r in range(0, len(features) + 1)))
# no features, all, and ablation
param_grid['add_features'] = list(chain.from_iterable(combinations(features, r) for r in [0, len(features) - 1, len(features)]))
# no features, all, and single
#param_grid['add_features'] = list(chain.from_iterable(combinations(features, r) for r in [0, len(features) + 1, 1]))
# no and all features
#param_grid['add_features'] = list(chain.from_iterable(combinations(features, r) for r in [0, len(features) + 1]))

param_grid['pkl_data_path'] = ['{}_{}_data.pkl']
param_grid['data_mode'] = ['full']

# run grid of options
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Usage: gppl task cut(percentage of training data to use) cut_mode')
        exit()

    task = sys.argv[1]
    cut = sys.argv[2] if len(sys.argv) > 2 else None
#    cut_mode = sys.argv[3] if len(sys.argv) > 3 else None
    experiment_dir = 'experiment_{}_{}'.format(task, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(experiment_dir)

    if task == 'metaphor':
        # only run on "original BWS" pairs for now
        param_grid['data_mode'] = ['pairs']
#        param_grid['data_mode'].append('pairs')

    # use only the best performing feature selection for cuts
    if cut is not None:
        if task == 'metaphor':
            param_grid['add_features'] = [('frequency', 'ngrams')]
        elif task == 'humour':
            param_grid['add_features'] = [('frequency', 'ngrams')]
        # use third-cuts
        param_grid['cut'] = [0.33, 0.66]
        param_grid['cut_mode'] = ['random_instances', 'random_pairs']

    grid = ParameterGrid(param_grid)
    logging.info('Starting [%d] experiments run to "%s"...', len(grid), experiment_dir)

    for OPTIONS in grid:

        random.seed(rnd_seed)
        np.random.seed(np_seed)

        cut = OPTIONS['cut']
        cut_mode = OPTIONS['cut_mode']

        if task == 'humour' and cut == 0.33 and cut_mode == 'random_instances':
            continue

        run(task, cut, cut_mode, experiment_dir)
