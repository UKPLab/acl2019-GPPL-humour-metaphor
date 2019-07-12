from tests import TestRunner

if __name__ == '__main__':

    acc = 1.0
    dataset_increment = 0

    # Classification and ranking tasks together

    datasets = ['UKPConvArgCrowdSample_evalMACE']
    methods = ['SVM', 'BI-LSTM', 'SinglePrefGP_noOpt_weaksprior',
               'SingleGPC_noOpt_weaksprior', 'GP+SVM']
    feature_types = ['both'] # ''
    embeddings_types = ['word_mean']

    runner = TestRunner('crowdsourcing_argumentation_expts', datasets, feature_types, embeddings_types, methods,
                        dataset_increment)
    runner.run_test_set()