'''
Simpler example showing how to use train and use the convincingness model for prediction.

This script trains a model on the UKPConvArgStrict dataset. So, before running this script, you need to run
"python/analysis/habernal_comparison/run_preprocessing.py" to extract the linguistic features from this dataset.

'''
import sys

# include the paths for the other directories
sys.path.append("./python")
sys.path.append("./python/analysis")
sys.path.append("./python/models")
sys.path.append("./python/analysis/habernal_comparison")

from data_loader import load_single_file_separate_args
from data_loading import load_ling_features, load_embeddings
from gp_classifier_vb import compute_median_lengthscales
from gp_pref_learning import GPPrefLearning
from run_preprocessing import preprocessing_pipeline
from tests import get_docidxs_from_ids, get_doc_token_seqs, get_mean_embeddings
import numpy as np
import logging
import os
from os import listdir
import vocabulary_embeddings_extractor
import pickle
import pandas as pd

# set the path for the java source code here

pkl_file = './model.pkl' # location to save the trained model to
test_data_path = './data/new_test_data' # location of your test data file. MUST HAVE A .CSV SUFFIX

embeddings_dir = './data/'

training_data_path = os.path.abspath("./data/")
training_dataset = 'UKPConvArgStrict'

def concat_feature_sets(a, X, ling_feat_spmatrix, embeddings, docid_to_idx_map=None):
    '''
    Combine linguistic features with mean embeddings.

    :param a:
    :param X:
    :param ling_feat_spmatrix:
    :param embeddings:
    :param docid_to_idx_map: a list of indexes into ling_feat_spmatrix sorted from low to high. Maps from
    the original document IDs to the indexes in the list of feature vectors.
    :return: list of feature vectors.
    '''
    X, u_ids = get_doc_token_seqs(a, X)
    items_feat = get_mean_embeddings(embeddings, X)

    print('No. documents with embeddings: %i' % items_feat.shape[0])

    if docid_to_idx_map is None:
        docid_to_idx_map = u_ids

    ling_items_feat = ling_feat_spmatrix.toarray()[docid_to_idx_map, :]

    items_feat = np.concatenate((items_feat, ling_items_feat), axis=1)

    items_feat[np.isnan(items_feat)] = 0

    print('Found %i features.' % items_feat.shape[1])

    return items_feat, u_ids

def load_train_dataset(dataset, embeddings):
    ling_feat_spmatrix, docids = load_ling_features(dataset, training_data_path)

    print('Number of documents with linguistic features: %i' % len(docids))

    data_root_dir = os.path.expanduser(training_data_path)
    csvdirname = os.path.join(data_root_dir, 'argument_data/%s-new-CSV/' % dataset)

    print(('Loading train/test data from %s...' % csvdirname))

    person_train = []
    a1_train = []
    a2_train = []
    ids_train = []
    prefs_train = []
    X_a1 = []
    X_a2 = []

    for file_name in listdir(csvdirname):
        if file_name.split('.')[-1] != 'csv':
            print("Skipping files without .csv suffix: %s" % csvdirname + '/' + file_name)
            continue

        Xa1, Xa2, labels, ids, turker_ids, a1, a2 = load_single_file_separate_args(csvdirname, file_name,
                                                                                  word_to_indices_map, None)

        X_a1.extend(Xa1)
        X_a2.extend(Xa2)

        a1_train.extend(a1)
        a2_train.extend(a2)

        person_train.extend(turker_ids)
        prefs_train.extend(labels)
        ids_train.extend(ids)

    train_ids = np.array([ids_pair.split('_') for ids_pair in ids_train])

    print('No. documents in training set: %i' % len(np.unique([train_ids[:, 0], train_ids[:, 1]])) )

    a1_train = get_docidxs_from_ids(docids, train_ids[:, 0])
    a2_train = get_docidxs_from_ids(docids, train_ids[:, 1])

    items_feat, uids = concat_feature_sets((a1_train, a2_train), [X_a1, X_a2], ling_feat_spmatrix, embeddings)

    ndims = items_feat.shape[1]

    return items_feat, ling_feat_spmatrix.shape[1], word_to_indices_map, a1_train, \
           a2_train, prefs_train, ndims

def train_model(embeddings):
    # Train a model...
    items_feat, n_ling_feats, word_to_indices_map, a1_train, a2_train, prefs_train, ndims \
       = load_train_dataset(training_dataset, embeddings)  # reload only if we use a new dataset

    ls_initial = compute_median_lengthscales(items_feat)

    model = GPPrefLearning(ninput_features=ndims, ls_initial=ls_initial, verbose=False,
                                shape_s0=2.0, rate_s0=200.0, rate_ls=1.0 / np.mean(ls_initial),
                                use_svi=True, ninducing=500, max_update_size=200, kernel_combination='*',
                                forgetting_rate=0.7, delay=1.0)

    model.max_iter_VB = 2000

    print("no. features: %i" % items_feat.shape[1])

    model.fit(a1_train, a2_train, items_feat, np.array(prefs_train, dtype=float) - 1, optimize=False,
              input_type='zero-centered')

    logging.info("**** Completed training GPPL ****")

    # Save the model in case we need to reload it

    with open(pkl_file, 'wb') as fh:
        pickle.dump(model, fh)

def load_test_dataset(output, embeddings):
    # Load the linguistic features
    print(("Loading linguistic features from %s" % output))
    ling_feat_spmatrix, docids = load_ling_features('new_test_data',
                       output,
                       '',
                       output,
                       model.features.shape[1] - len(embeddings[0])
                       )

    print('Loaded libSVM data')

    X = []
    test_ids = []
    a = []

    for file_name in listdir(input_dir):
        if file_name.split('.')[-1] != 'csv':
            print("Skipping files without .csv suffix: %s" % input_dir + '/' + file_name)
            continue

        data = pd.read_csv(os.path.join(input_dir, file_name), delimiter='\t', na_values=[])
        data = data.fillna('N/A')

        ids = data['#id'].values
        a1 = data['argument'].values

        a1_tokens = [vocabulary_embeddings_extractor.tokenize(a1_line) for a1_line in a1]
        a1_indices = [[word_to_indices_map.get(word, 2) for word in a1_tokens_line] for a1_tokens_line in a1_tokens]
        Xa1 = np.array([[1] + a1_indices_line for a1_indices_line in a1_indices])

        valid_args = np.in1d(ids, docids)
        a1 = a1[valid_args]
        Xa1 = Xa1[valid_args]
        ids = ids[valid_args]

        a.extend(a1)
        X.extend(Xa1)
        test_ids.extend(ids)

    # load the embeddings
    docid_to_idx_map = np.argsort(docids).flatten()
    test_items_feat, uids = concat_feature_sets((test_ids), [X], ling_feat_spmatrix, embeddings, docid_to_idx_map)

    return test_items_feat, uids

if __name__ == '__main__':

    print('This script trains a model on the UKPConvArgStrict dataset. So, before running this script, you '
          'need to run "python/analysis/habernal_comparison/run_preprocessing.py" to extract the linguistic features'
          'from this dataset.')

    word_to_indices_map, word_index_to_embeddings_map, index_to_word_map = vocabulary_embeddings_extractor.load_all(
        embeddings_dir + 'vocabulary.embeddings.all.pkl.bz2')
    embeddings = load_embeddings(word_index_to_embeddings_map)

    train_model(embeddings)

    # Load the model and the embeddings from file
    with open(pkl_file, 'rb') as fh:
        model = pickle.load(fh)

    # Now load some test documents for RANKING and extract their features
    input_dir = os.path.abspath(test_data_path)
    tmp_dir = os.path.abspath('./data/tempdata')
    output_dir = os.path.abspath('./data/new_ranking_libsvm')

    # use this directory to get a mapping from features to integers that matches the training set
    feature_dir = os.path.join(os.path.expanduser(training_data_path), 'tempdata/UKPConvArg1-full3')

    preprocessing_pipeline(input_dir, output_dir, 'new_test_ranking', tmp_dir, feature_dir, remove_tabs=True)

    test_items_feat, text_ids = load_test_dataset(output_dir, embeddings)

    print('Predicting ...')
    predicted_f, _ = model.predict_f(out_feats=test_items_feat)

    print('Results: id, score ')
    for i in range(len(text_ids)):
        print('%s, %s' % (text_ids[i], predicted_f[i]))