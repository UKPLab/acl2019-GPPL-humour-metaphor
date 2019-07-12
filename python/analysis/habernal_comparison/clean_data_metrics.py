import os

from compute_metrics import compute_metrics

if __name__ == '__main__':
    
    if 'expt_settings' not in globals():
        expt_settings = {}
        expt_settings['dataset'] = None
        expt_settings['folds'] = None

    expt_settings['foldorderfile'] = None

    npairs = 0
    acc = 1.0
    di = 0

    max_no_folds = 32

    # Classification tasks

    print('*** Performance metrics for UKPConvArgStrict, ling features ***')

    datasets = ['UKPConvArgStrict']
    methods = ['SVM', 'SinglePrefGP_noOpt_weaksprior']
    feature_types = ['ling']
    embeddings_types = ['word_mean']

    results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
    tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
    = compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs,
                      max_no_folds=max_no_folds)

    print('*** Performance metrics for UKPConvArgStrict, embeddings features ***')

    datasets = ['UKPConvArgStrict']
    methods = ['BI-LSTM', 'SinglePrefGP_noOpt_weaksprior']
    feature_types = ['embeddings']
    embeddings_types = ['word_mean']


    results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
    tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
    = compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs,
                      max_no_folds=max_no_folds)

    print('*** Performance metrics for UKPConvArgStrict, ling+Glove features ***')

    datasets = ['UKPConvArgStrict']
    methods = ['SVM', 'BI-LSTM', 'SinglePrefGP_noOpt_weaksprior', 'SinglePrefGP_weaksprior',
               'SingleGPC_noOpt_weaksprior', 'GP+SVM']
    feature_types = ['both']
    embeddings_types = ['word_mean']

    results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
    tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
    = compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs,
                      max_no_folds=max_no_folds)

    # Ranking tasks

    print('*** Performance metrics for UKPConvArgAll, ling features ***')

    datasets = ['UKPConvArgAll']
    methods = ['SVM', 'SinglePrefGP_noOpt_weaksprior']
    feature_types = ['ling']
    embeddings_types = ['word_mean']

    results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
    tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
    = compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs,
                      max_no_folds=max_no_folds)


    print('*** Performance metrics for UKPConvArgAll, embeddings features ***')

    datasets = ['UKPConvArgAll']
    methods = ['BI-LSTM', 'SinglePrefGP_noOpt_weaksprior']
    feature_types = ['embeddings']
    embeddings_types = ['word_mean']


    results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
    tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
    = compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs,
                      max_no_folds=max_no_folds)

    print('*** Performance metrics for UKPConvArgAll, ling+Glove features ***')

    datasets = ['UKPConvArgAll']
    methods = ['SVM', 'BI-LSTM', 'SinglePrefGP_noOpt_weaksprior', 'SinglePrefGP_weaksprior',
               'SingleGPC_noOpt_weaksprior', 'GP+SVM']
    feature_types = ['both']
    embeddings_types = ['word_mean']

    results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
    tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
    = compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs,
                      max_no_folds=max_no_folds)


    print("Completed compute metrics")

