'''
Created on 24 Aug 2017

@author: simpson

Error analysis steps for preference learning for convincingness paper.

'''
import sys
import os

sys.path.append("./python")
sys.path.append("./python/analysis")
sys.path.append("./python/models")
sys.path.append("./python/analysis/habernal_comparison")
svm_python_path = '~/libsvm-3.22/python'

sys.path.append(os.path.expanduser("~/git/HeatMapBCC/python"))
sys.path.append(os.path.expanduser("~/git/pyIBCC/python"))

sys.path.append(os.path.expanduser("~/data/personalised_argumentation/embeddings/skip-thoughts"))
sys.path.append(os.path.expanduser("~/data/personalised_argumentation/embeddings/Siamese-CBOW/siamese-cbow"))

sys.path.append(os.path.expanduser(svm_python_path))

import pickle
import numpy as np

from compute_metrics import get_results_dir, get_fold_data
from data_loading import load_train_test_data, load_ling_features
from tests import get_doc_token_seqs, get_docidxs_from_ids, compute_lengthscale_heuristic
from gp_classifier_vb import matern_3_2_from_raw_vals
from sklearn.metrics.pairwise import cosine_similarity

def get_text_from_fold_regression(fold, dataset):
    X_test, _, ids_test, _, test_a = fold

    # Identify the arguments in the false pairs.
    X_test = np.array(X_test)

    _, docids = load_ling_features(dataset)
    testids_a = get_docidxs_from_ids(docids, ids_test)
    _, _, utexts = get_doc_token_seqs((testids_a), [X_test], [test_a])

    return testids_a, utexts

def get_text_from_fold(fold, dataset):
    X_test_a1, X_test_a2, _, ids_test, _, test_a1, test_a2 = fold
    test_a1 = np.array(test_a1).flatten()
    test_a2 = np.array(test_a2).flatten()

    # Identify the arguments in the false pairs.
    X_test_a1 = np.array(X_test_a1)
    X_test_a2 = np.array(X_test_a2)

    testids = np.array([ids_pair.split('_') for ids_pair in ids_test])

    _, docids = load_ling_features(dataset)
    testids_a1 = get_docidxs_from_ids(docids, testids[:, 0])
    testids_a2 = get_docidxs_from_ids(docids, testids[:, 1])
    _, _, utexts = get_doc_token_seqs((testids_a1, testids_a2), [X_test_a1, X_test_a2], (test_a1, test_a2))

    return testids_a1, testids_a2, utexts

def print_where_one_right_two_wrong(expt_settings, feature_type_one, feature_type_two, embeddings_type_one,
                                    embeddings_type_two, method1=None, method2=None):

    # Load the results for GPPL with Ling.
    expt_settings_1 = expt_settings.copy()
    expt_settings_1['feature_type'] = feature_type_one
    expt_settings_1['embeddings_type'] = embeddings_type_one
    if method1 is not None:
        expt_settings_1['method'] = method1

    data_root_dir = os.path.expanduser("~/data/personalised_argumentation/")
    resultsfile_template = 'habernal_%s_%s_%s_%s_acc%.2f_di%.2f'

    resultsdir_1 = get_results_dir(data_root_dir, resultsfile_template, expt_settings_1)

    # Load the results for GPPL with Glove.
    expt_settings_2 = expt_settings.copy()
    expt_settings_2['feature_type'] = feature_type_two
    expt_settings_2['embeddings_type'] = embeddings_type_two
    if method2 is not None:
        expt_settings_2['method'] = method2
    resultsdir_2 = get_results_dir(data_root_dir, resultsfile_template, expt_settings_2)

    nFolds = len(list(folds.keys()))

    allcounts_1 = np.zeros(0)
    alltexts_1 = np.zeros(0)

    # count up the types of mistake for each argument.
    # If the argument is a1, +1 means label should be 0 but 2 was predicted (argument was overrated by the algorithm).
    # -1 means label should be 2 but +1 was predicted (argument was underrated)
    # If the argument is a2, flip the labels.
    errorcounts_1 = np.zeros((0, 2)) # first column for +1s and second column for -1s

    for f in range(nFolds):
        fold = list(folds.keys())[f]

        if 'fold_order' in expt_settings_1 and expt_settings_1['fold_order'] is not None:
            f1 = np.argwhere(np.array(expt_settings_1['fold_order']) == fold)[0][0]
        else:
            f1 = f

        foldfile = resultsdir_1 + '/fold%i.pkl' % f1
        if os.path.isfile(foldfile):
            with open(foldfile, 'rb') as fh:
                data_1 = pickle.load(fh, encoding='latin1')

        if 'fold_order' in expt_settings_2 and expt_settings_2['fold_order'] is not None:
            f2 = np.argwhere(np.array(expt_settings_2['fold_order']) == fold)[0][0]
        else:
            f2 = f

        foldfile = resultsdir_2 + '/fold%i.pkl' % f2
        if os.path.isfile(foldfile):
            with open(foldfile, 'rb') as fh:
                data_2 = pickle.load(fh, encoding='latin1')

        # Load the ground truth classifications
        gold_disc_1, pred_disc_1, _, _, _, _, _, _, _ = get_fold_data(data_1, f1, expt_settings_1)
        gold_disc_2, pred_disc_2, _, _, _, _, _, _, _ = get_fold_data(data_2, f2, expt_settings_2)

        # Identify the falsely classified pairs with Ling
        #gold_disc_1 = gold_disc_1[:, None]
        #gold_disc_2 = gold_disc_2[:, None]
        if expt_settings_1['method'] == 'SVM':
            pred_disc_1 = pred_disc_1[:, 1]
        if expt_settings_2['method'] == 'SVM':
            pred_disc_2 = pred_disc_2[:, 1]
        pred_disc_1 = pred_disc_1.flatten()
        pred_disc_2 = pred_disc_2.flatten()
        false_pairs_1 = pred_disc_1 != gold_disc_1
        false_pairs_2 = pred_disc_2 != gold_disc_2

        errors1 = np.zeros((len(pred_disc_1), 2), dtype=int)
        errors1[:, 0] = (gold_disc_1==0) & (pred_disc_1==2) # a1 was underrated, a2 was overrated
        errors1[:, 1] = -((gold_disc_1==2) & (pred_disc_1==0)).astype(int) # a1 was overrated, a2 was underrated

        errors2 = np.zeros((len(pred_disc_2), 2), dtype=int)
        errors2[:, 0] = (gold_disc_2==0) & (pred_disc_2==2)
        errors2[:, 1] = -((gold_disc_2==2) & (pred_disc_2==0)).astype(int)

        # pairs falsely classified by one and not the other
        if len(false_pairs_1) != len(false_pairs_2):
            print(("mismatched fold %i: " % f))
            print(("word mean: " + str(len(false_pairs_1))))
            print(("ling: " + str(len(false_pairs_2))))
            print((len(folds[fold]['test'][0])))
            continue

        # cases where 1 is wrong and 2 is right
        false_pairs_only1 = false_pairs_1 & np.invert(false_pairs_2)
        errors1 = errors1[false_pairs_only1]

        # Get the argument IDs for this fold
        #X_train_a1, X_train_a2 are lists of lists of word indexes
        #X_train_a1, X_train_a2, prefs_train, ids_train, personIDs_train, tr_a1, tr_a2 = folds.get(fold)["training"]
        testids_a1, testids_a2, utexts = get_text_from_fold(folds.get(fold)['test'], expt_settings_1['dataset'])

        # Rank by number of false pairwise labels.
        falseids_1only_a1 = testids_a1[false_pairs_only1]
        falseids_1only_a2 = testids_a2[false_pairs_only1]
        counts = np.bincount(np.concatenate((falseids_1only_a1, falseids_1only_a2)))
        error_sum1 = np.zeros((len(counts), 2))
        for i, argid in enumerate(falseids_1only_a1):
            errors_arg1 = errors1[falseids_1only_a1==argid, :]
            error_sum1[argid, :] += np.sum(errors_arg1, axis=0)

        for i, argid in enumerate(falseids_1only_a2):
            errors_arg2 = errors1[falseids_1only_a2==argid, :]

            error_sum1[argid, 0] -= np.sum(errors_arg2, axis=0)[1] # flip the order
            error_sum1[argid, 1] -= np.sum(errors_arg2, axis=0)[0]
        top = np.argsort(counts)[-25:]

        allcounts_1 = np.concatenate((allcounts_1, counts[top]))
        false_texts = np.array(utexts)[top]
        alltexts_1 = np.concatenate((alltexts_1, false_texts))
        errorcounts_1 = np.concatenate((errorcounts_1, error_sum1[top]))

        top = np.argsort(counts)[-25:]

    # Pick the top 25.
    top_ids = np.argsort(allcounts_1)[-25:]
    top_texts_1 = alltexts_1[top_ids]

    # Get the original argument text. Print it with the mis-classification label and score.
    print(("Showing the top 25 most incorrectly labelled arguments for %s: " % expt_settings_1['feature_type']))
    print("No. times underrated; no.times overrated; argument text")
    for i, text in enumerate(top_texts_1):
        print(("\t %i %i %s \n" % (errorcounts_1[i, 0], errorcounts_1[i, 1], text)))

def compute_max_train_similarity(expt_settings, method, ls, docids, items_feat, similarities_all=None):
    '''
    Find the maximum cosine similarity for arguments in the dataset.

    Compute the mean/variance of the max similarity for correct/incorrect pairs.
    '''
    # Load the results for GPPL with Ling.
    expt_settings_1 = expt_settings.copy()
    expt_settings_1['method'] = method
    expt_settings_1['feature_type'] = 'ling'
    expt_settings_1['embeddings_type'] = ''

    data_root_dir = os.path.expanduser("~/data/personalised_argumentation/")
    resultsfile_template = 'habernal_%s_%s_%s_%s_acc%.2f_di%.2f'

    resultsdir_1 = get_results_dir(data_root_dir, resultsfile_template, expt_settings_1)

    # Load the results for GPPL with Glove.

    nFolds = len(list(folds.keys()))

    mean_false = 0
    mean_true = 0
    var_false = 0
    var_true = 0

    if similarities_all is None:
        feats = items_feat / ls[None, :]
        similarities_all = cosine_similarity(feats, feats, dense_output=True)#matern_3_2_from_raw_vals(items_feat, ls, items_feat)

    total_count_true = 0
    total_count_false = 0

    for f in range(nFolds):
        fold = list(folds.keys())[f]

        if 'fold_order' in expt_settings_1 and expt_settings_1['fold_order'] is not None:
            f1 = np.argwhere(np.array(expt_settings_1['fold_order']) == fold)[0][0]
        else:
            f1 = f

        foldfile = resultsdir_1 + '/fold%i.pkl' % f1
        if os.path.isfile(foldfile):
            with open(foldfile, 'rb') as fh:
                data_1 = pickle.load(fh, encoding='latin1')

        # Load the ground truth classifications
        gold_disc_1, pred_disc_1, _, _, _, _, _, _, _ = get_fold_data(data_1, f1, expt_settings_1)

        # Identify the falsely classified pairs with Ling
        #gold_disc_1 = gold_disc_1[:, None]
        #gold_disc_2 = gold_disc_2[:, None]
        if expt_settings_1['method'] == 'SVM':
            pred_disc_1 = pred_disc_1[:, 1]
        pred_disc_1 = pred_disc_1.flatten()
        false_pairs_1 = pred_disc_1 != gold_disc_1
        true_pairs_1 = pred_disc_1 == gold_disc_1

        # Get the argument IDs for this fold
        X_test_a1, X_test_a2, _, ids_test, _, test_a1, test_a2 = folds.get(fold)["test"]
        test_a1 = np.array(test_a1)[:, None]
        test_a2 = np.array(test_a2)[:, None]

        testids = np.array([ids_pair.split('_') for ids_pair in ids_test])
        X_test_a1 = np.array(X_test_a1)
        X_test_a2 = np.array(X_test_a2)

        testids_a1 = get_docidxs_from_ids(docids, testids[:, 0])
        testids_a2 = get_docidxs_from_ids(docids, testids[:, 1])

        X_tr_a1, X_tr_a2, _, ids_tr, _, tr_a1, tr_a2 = folds.get(fold)["training"]
        tr_a1 = np.array(tr_a1)[:, None]
        tr_a2 = np.array(tr_a2)[:, None]

        trids = np.array([ids_pair.split('_') for ids_pair in ids_tr])
        X_tr_a1 = np.array(X_tr_a1)
        X_tr_a2 = np.array(X_tr_a2)

        _, docids = load_ling_features(expt_settings_1['dataset'])
        trids_a1 = get_docidxs_from_ids(docids, trids[:, 0])
        trids_a2 = get_docidxs_from_ids(docids, trids[:, 1])

        true_similarities = similarities_all[np.concatenate((testids_a1[true_pairs_1], testids_a2[true_pairs_1])), :]\
                                [:, np.concatenate((trids_a1, trids_a2))]
        true_similarities = np.max(true_similarities, axis=1)

        false_similarities = similarities_all[np.concatenate((testids_a1[false_pairs_1], testids_a2[false_pairs_1])), :]\
                                [:, np.concatenate((trids_a1, trids_a2))]
        false_similarities = np.max(false_similarities, axis=1)

        total_count_true += np.sum(true_pairs_1) * 2.0
        total_count_false += np.sum(false_pairs_1) * 2.0
        mean_total_sims_true = np.sum(true_similarities)
        mean_total_sims_false = np.sum(false_similarities)
        var_total_sims_true  = np.var(true_similarities)
        var_total_sims_false = np.var(false_similarities)

        mean_false += mean_total_sims_false
        mean_true += mean_total_sims_true
        var_false += var_total_sims_false
        var_true += var_total_sims_true

        #print "mean total_similarity for correctly classified pairs: %f (STD %f)" % (mean_total_sims_true,
        #                                                                           np.sqrt(var_total_sims_true))
        #print "mean total_similarity for incorrectly classified pairs: %f (STD %f)" % (mean_total_sims_false,
        #                                                                            np.sqrt(var_total_sims_false))
        sys.stdout.write('.'); sys.stdout.flush()

    mean_false /= total_count_false
    mean_true /= total_count_true
    var_false /= nFolds
    var_false = np.sqrt(var_false)
    var_true /= nFolds
    var_true = np.sqrt(var_true)

    print(("For all folds: mean total_sim for correctly classified pairs: %f (STD %f)" % (mean_true,
                                                                                          np.sqrt(var_true))))
    print(("For all folds: mean total_sim for incorrectly classified pairs: %f (STD %f)" % (mean_false,
                                                                                        np.sqrt(var_false))))

    return similarities_all

def compute_entropies(expt_settings, method, feature_type, embeddings_type):
    expt_settings_1 = expt_settings.copy()
    expt_settings_1['method'] = method
    expt_settings_1['feature_type'] = feature_type
    expt_settings_1['embeddings_type'] = embeddings_type

    data_root_dir = os.path.expanduser("~/data/personalised_argumentation/")
    resultsfile_template = 'habernal_%s_%s_%s_%s_acc%.2f_di%.2f'

    resultsdir_1 = get_results_dir(data_root_dir, resultsfile_template, expt_settings_1)

    nFolds = len(list(folds.keys()))

    true_entropy = np.zeros(0)
    false_entropy = np.zeros(0)

    for f in range(nFolds):
        fold = list(folds.keys())[f]

        if 'fold_order' in expt_settings_1 and expt_settings_1['fold_order'] is not None:
            f1 = np.argwhere(np.array(expt_settings_1['fold_order']) == fold)[0][0]
        else:
            f1 = f

        foldfile = resultsdir_1 + '/fold%i.pkl' % f1
        if os.path.isfile(foldfile):
            with open(foldfile, 'rb') as fh:
                data_1 = pickle.load(fh, encoding='latin1')
        else:
            print(("Error -- data not found at %s" % foldfile))

        gold_disc_1, pred_disc_1, gold_prob, pred_prob, _, _, _, _, _ = get_fold_data(data_1, f1, expt_settings_1)
        if expt_settings_1['method'] == 'SVM':
            if pred_disc_1.ndim == 2:
                pred_disc_1 = pred_disc_1[:, 1]
            if pred_prob.ndim == 2:
                pred_prob = pred_prob[:, 1]

        pred_prob = pred_prob.flatten()
        entropy = - gold_prob * np.log(pred_prob) - (1 - gold_prob) * np.log(1 - pred_prob)

        pred_disc_1 = pred_disc_1.flatten()
        false_pairs_1 = pred_disc_1 != gold_disc_1
        true_pairs_1 = pred_disc_1 == gold_disc_1

        false_entropy_f = entropy[false_pairs_1]
        true_entropy_f = entropy[true_pairs_1]

        false_entropy = np.concatenate((false_entropy, false_entropy_f))
        true_entropy = np.concatenate((true_entropy, true_entropy_f))

    return np.mean(true_entropy), np.mean(false_entropy)

def compute_errors_in_training(expt_settings, method, feature_type, embeddings_type):

    expt_settings_1 = expt_settings.copy()
    expt_settings_1['dataset'] = 'UKPConvArgCrowdSample_evalMACE'
    folds_noisy, _, _, _, _ = load_train_test_data(expt_settings_1['dataset'])
    expt_settings_1['folds'] = folds_noisy

    expt_settings_2 = expt_settings.copy()
    expt_settings_2['dataset'] = 'UKPConvArgAll'
    folds_clean, _, _, _, _ = load_train_test_data(expt_settings['dataset'])

    expt_settings_1['method'] = method
    expt_settings_1['feature_type'] = feature_type
    expt_settings_1['embeddings_type'] = embeddings_type

    data_root_dir = os.path.expanduser("~/data/personalised_argumentation/")
    resultsfile_template = 'habernal_%s_%s_%s_%s_acc%.2f_di%.2f'

    resultsdir_1 = get_results_dir(data_root_dir, resultsfile_template, expt_settings_1)

    nFolds = len(list(folds_noisy.keys()))

    correct_given_trerror = np.zeros((2, 2)) # correct/incorrect answers (cols) given training label errors (rows)

    for f in range(nFolds):
        fold = list(folds_noisy.keys())[f]

        if 'fold_order' in expt_settings_1 and expt_settings_1['fold_order'] is not None:
            f1 = np.argwhere(np.array(expt_settings_1['fold_order']) == fold)[0][0]
        else:
            f1 = f

        foldfile = resultsdir_1 + '/fold%i.pkl' % f1
        if os.path.isfile(foldfile):
            with open(foldfile, 'rb') as fh:
                data_1 = pickle.load(fh, encoding='latin1')
        else:
            print(("Error -- data not found at %s" % foldfile))

        _, _, _, _, _, _, pred_tr, _, _ = get_fold_data(data_1, f1, expt_settings_1)
        if expt_settings_1['method'] == 'SVM':
            if pred_tr.ndim == 2:
                pred_tr = pred_tr[:, 1]
        pred_tr = pred_tr.flatten()

        # find the errors in the training data
        noisy_gold_tr = folds_noisy[fold]['training'][2]
        clean_gold_tr = folds_clean[fold]['training'][2]

        trerrors = noisy_gold_tr != clean_gold_tr
        trcorrect = noisy_gold_tr == clean_gold_tr

        pred_errors = pred_tr != clean_gold_tr
        pred_correct = pred_tr == clean_gold_tr

        correct_given_trerror[0, 0] += np.sum(pred_correct & trcorrect) # no training error, correct prediction
        correct_given_trerror[0, 1] += np.sum(pred_errors & trcorrect) # no training error, incorrect prediction
        correct_given_trerror[1, 0] += np.sum(pred_correct & trerrors) # training error, correct prediction
        correct_given_trerror[1, 1] += np.sum(pred_errors & trerrors) # training error, incorrect prediction

    print("Matrix of counts of correct predictions given training data errors:")
    print(correct_given_trerror)

    return correct_given_trerror

def print_best_worst(expt_settings, method, feature_type, embeddings_type, folds_r):
    # Load the results for GPPL with Ling.
    expt_settings_1 = expt_settings.copy()
    expt_settings_1['method'] = method
    expt_settings_1['feature_type'] = feature_type
    expt_settings_1['embeddings_type'] = embeddings_type

    data_root_dir = os.path.expanduser("~/data/personalised_argumentation/")
    resultsfile_template = 'habernal_%s_%s_%s_%s_acc%.2f_di%.2f'

    resultsdir_1 = get_results_dir(data_root_dir, resultsfile_template, expt_settings_1)

    nFolds = len(list(folds.keys()))

    for f in range(nFolds):
        fold = list(folds.keys())[f]

        if 'fold_order' in expt_settings_1 and expt_settings_1['fold_order'] is not None:
            f1 = np.argwhere(np.array(expt_settings_1['fold_order']) == fold)[0][0]
        else:
            f1 = f

        foldfile = resultsdir_1 + '/fold%i.pkl' % f1
        if os.path.isfile(foldfile):
            with open(foldfile, 'rb') as fh:
                data_1 = pickle.load(fh, encoding='latin1')
        else:
            print(("Error -- data not found at %s" % foldfile))

        # Load the ground truth classifications
        _, _, _, _, gold_score, _, _, _, _ = get_fold_data(data_1, f1, expt_settings_1)

        # Sort
        gold_ranked_idxs = np.argsort(gold_score)

        # get associated text
        argids, utexts_f = get_text_from_fold_regression(folds_r.get(fold)['test'], expt_settings_1['dataset'])
        utexts_f = np.array(utexts_f)[argids]

        topargs = gold_ranked_idxs[-10:]
        bottomargs = gold_ranked_idxs[:10]

        for arg in topargs:
            print((utexts_f[arg]))

        for arg in bottomargs:
            print((utexts_f[arg]))

def get_rank_deviations(expt_settings, method, feature_type, embeddings_type, folds_r):
    # Load the results for GPPL with Ling.
    expt_settings_1 = expt_settings.copy()
    expt_settings_1['method'] = method
    expt_settings_1['feature_type'] = feature_type
    expt_settings_1['embeddings_type'] = embeddings_type

    data_root_dir = os.path.expanduser("~/data/personalised_argumentation/")
    resultsfile_template = 'habernal_%s_%s_%s_%s_acc%.2f_di%.2f'

    resultsdir_1 = get_results_dir(data_root_dir, resultsfile_template, expt_settings_1)

    deviations = np.zeros(0)
    utexts = np.zeros(0)

    nFolds = len(list(folds.keys()))

    for f in range(nFolds):
        fold = list(folds.keys())[f]

        if 'fold_order' in expt_settings_1 and expt_settings_1['fold_order'] is not None:
            f1 = np.argwhere(np.array(expt_settings_1['fold_order']) == fold)[0][0]
        else:
            f1 = f

        foldfile = resultsdir_1 + '/fold%i.pkl' % f1
        if os.path.isfile(foldfile):
            with open(foldfile, 'rb') as fh:
                data_1 = pickle.load(fh, encoding='latin1')
        else:
            print(("Error -- data not found at %s" % foldfile))

        # Load the ground truth classifications
        _, _, _, _, gold_score, pred_score, _, _, _ = get_fold_data(data_1, f1, expt_settings_1)
        pred_score = pred_score.flatten()

        # Sort
        gold_ranked_idxs = np.argsort(gold_score)
        pred_ranked_idxs = np.argsort(pred_score)

        deviations_f = np.array([i - np.argwhere(pred_ranked_idxs==idx)[0][0] for i, idx in enumerate(gold_ranked_idxs)])

        deviations = np.concatenate((deviations, deviations_f))

        # get associated text
        argids, utexts_f = get_text_from_fold_regression(folds_r.get(fold)['test'], expt_settings_1['dataset'])
        utexts_f = np.array(utexts_f)[argids]

        utexts = np.concatenate((utexts, utexts_f))

    return deviations, utexts

def print_arg_text(errors_1, errors_2, top_texts):
    for i, text in enumerate(top_texts):
        print(("\t %i %i %s \n" % (errors_1[i], errors_2[i], text)))

if __name__ == '__main__':
    # Step 1. Inspecting arguments in fifty pairs falsely classified by GPPL using only Glove or only ling features.
    if 'expt_settings' not in globals():
        expt_settings = {}
        expt_settings['acc'] = 1.0
        expt_settings['di'] = 0.0
        expt_settings['foldorderfile'] = None

    expt_settings['dataset'] = 'UKPConvArgStrict'
    expt_settings['method'] = 'SinglePrefGP_noOpt_weaksprior'
    folds, folds_regression, _, _, _ = load_train_test_data(expt_settings['dataset'])
    expt_settings['folds'] = folds
    expt_settings['folds_regression'] = folds_regression


    print_where_one_right_two_wrong(expt_settings, 'embeddings', 'ling', 'word_mean', '')
    print_where_one_right_two_wrong(expt_settings, 'ling', 'embeddings', '', 'word_mean')

    # step 2. Inspect the arguments that 'both' gets right and embeddings or ling alone gets wrong. Expect the results
    # to be similar to the same as the previous step.
    print_where_one_right_two_wrong(expt_settings, 'embeddings', 'both', 'word_mean', 'word_mean')
    print_where_one_right_two_wrong(expt_settings, 'ling', 'both', '', 'word_mean')

    print_where_one_right_two_wrong(expt_settings, 'both', 'embeddings', 'word_mean', 'word_mean')
    print_where_one_right_two_wrong(expt_settings, 'both', 'ling', 'word_mean', '')

    # Step 3: Compare GPPL to SVM to see which handles outliers better given same features
    ling_feat_spmatrix, docids = load_ling_features(expt_settings['dataset'])

    if 'ls' not in globals():
        ls = compute_lengthscale_heuristic('ling', '', None, ling_feat_spmatrix, docids, folds, None,
                                       multiply_heuristic_power=0.5)
    items_feat = ling_feat_spmatrix.toarray()

    if 'similarity' not in globals():
        similarity = None
    similarity = compute_max_train_similarity(expt_settings, 'SinglePrefGP_noOpt_weaksprior', ls, docids, items_feat, similarity)
    similarity = compute_max_train_similarity(expt_settings, 'SVM', ls, docids, items_feat, similarity)

    # Step 4: Compare GPPL to SVM look for other patterns in different errors given best features available
    print_where_one_right_two_wrong(expt_settings, 'both', 'ling', 'word_mean', '', 'SinglePrefGP_weaksprior', 'SVM')
    print_where_one_right_two_wrong(expt_settings, 'ling', 'both', '', 'word_mean', 'SVM', 'SinglePrefGP_weaksprior')

    # Step 8: Compute entropy for classifier predictions by SVM and GPPL for false/true labels; ling features for both
    print("Entropy for SVM correct/incorrect labels: %f, %f" % compute_entropies(expt_settings, 'SVM', 'ling', ''))
    print("Entropy for GPPL correct/incorrect labels: %f, %f" % compute_entropies(expt_settings,
                                                                      'SinglePrefGP_weaksprior', 'both', 'word_mean'))

    # a. Compute the rankings for GPPL and SVM.
    # b. Compute deviations from gold rank.
    expt_settings['dataset'] = 'UKPConvArgAll'
    folds, folds_regression, _, _, _ = load_train_test_data(expt_settings['dataset'])
    expt_settings['folds'] = folds
    expt_settings['folds_regression'] = folds_regression

    print("Best and worst args from each topic: ")
    print_best_worst(expt_settings, 'SinglePrefGP_weaksprior', 'both', 'word_mean', folds_regression)
    print("--------")

    deviations_gppl, utexts_gppl = get_rank_deviations(expt_settings, 'SinglePrefGP_weaksprior', 'both', 'word_mean',
                                                       folds_regression)
    deviations_svm, utexts_svm = get_rank_deviations(expt_settings, 'SVM', 'ling', '', folds_regression)

    # c. Compute differences in deviations between SVM and GPPL
    diffs = np.abs(deviations_svm) - np.abs(deviations_gppl)
    diff_args = np.argsort(diffs)

    # Step 5: Ranking errors that SVM made but GPPL solves - Look at top 20 diffs in deviations
    argidxs = diff_args[-20:]
    # load the matching argument texts
    print("Biggest differences in rank deviation between SVM and GPPL: where SVM was worse")
    print("Rank deviations by gppl; rank deviations by SVM; argument text")
    top_texts = utexts_svm[argidxs]
    print_arg_text(deviations_gppl[argidxs], deviations_svm[argidxs], top_texts)

    # Step 6: Ranking errors that GPPL makes the SVM did not make - Look at top 20 negative diffs in deviations. Use best version of GPPL.
    argidxs = diff_args[:20]
    print("Biggest differences in rank deviation between SVM and GPPL: where GPPL was worse")
    print("Rank deviations by gppl; rank deviations by SVM; argument text")
    top_texts = utexts_gppl[argidxs]
    print_arg_text(deviations_gppl[argidxs], deviations_svm[argidxs], top_texts)

    # Step 7: Other ranking errors that GPPL still makes - Look at top 20 deviations for GPPL.
    dev_args = np.argsort(np.abs(deviations_gppl))
    argidxs = dev_args[-20:]
    print("Biggest rank deviations for GPPL only")
    print("Rank deviations by gppl; -; argument text")
    top_texts = utexts_gppl[argidxs]
    print_arg_text(deviations_gppl[argidxs], np.zeros(len(deviations_gppl)), top_texts)


