import os

from compute_metrics import compute_metrics

if __name__ == '__main__':
    
    if 'expt_settings' not in globals():
        expt_settings = {}
        expt_settings['dataset'] = None
        expt_settings['folds'] = None

    expt_settings['foldorderfile'] = None
    data_root_dir = os.path.expanduser("~/data/personalised_argumentation/")

    resultsfile_template = 'habernal_%s_%s_%s_%s_acc%.2f_di%.2f'
    max_no_folds = 32

    npairs = 0
    acc = 1.0
    di = 0

    # Classification and ranking tasks together

    datasets = ['UKPConvArgCrowdSample_evalMACE']
    methods = ['SVM', 'BI-LSTM', 'SinglePrefGP_noOpt_weaksprior',
               'SingleGPC_noOpt_weaksprior', 'GP+SVM']
    feature_types = ['debug'] # 'both'
    embeddings_types = ['word_mean']

    results_f1, results_acc, results_auc, results_logloss, results_pearson, results_spearman, results_kendall, \
    tr_results_f1, tr_results_acc, tr_results_auc, tr_results_logloss, mean_results, combined_labels \
    = compute_metrics(expt_settings, methods, datasets, feature_types, embeddings_types, di=di, npairs=npairs,
                      max_no_folds=max_no_folds)

    print("Completed compute metrics")

