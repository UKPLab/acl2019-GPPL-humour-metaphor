from tests import TestRunner

if __name__ == '__main__':

    acc = 1.0
    dataset_increment = 0

    # Classifications tasks

    datasets = ['UKPConvArgStrict']
    methods = ['SVM', 'SinglePrefGP_noOpt_weaksprior']
    feature_types = ['ling'] # ''
    embeddings_types = ['word_mean']

    runner = TestRunner('crowdsourcing_argumentation_expts', datasets, feature_types, embeddings_types, methods,
                        dataset_increment)
    runner.run_test_set()

    datasets = ['UKPConvArgStrict']
    methods = ['BI-LSTM', 'SinglePrefGP_noOpt_weaksprior']
    feature_types = ['embeddings'] # ''
    embeddings_types = ['word_mean']

    runner = TestRunner('crowdsourcing_argumentation_expts', datasets, feature_types, embeddings_types, methods,
                        dataset_increment)
    runner.run_test_set()

    datasets = ['UKPConvArgStrict']
    methods = ['SinglePrefGP_noOpt_weaksprior', 'SVM', 'BI-LSTM', 'SinglePrefGP_weaksprior',
               'SingleGPC_noOpt_weaksprior', 'GP+SVM']
    feature_types = ['both'] #
    embeddings_types = ['word_mean']

    runner = TestRunner('crowdsourcing_argumentation_expts', datasets, feature_types, embeddings_types, methods,
                        dataset_increment)
    runner.run_test_set()

    # Ranking tasks

    datasets = ['UKPConvArgAll']
    methods = ['SVM', 'SinglePrefGP_noOpt_weaksprior']
    feature_types = ['ling'] # ''
    embeddings_types = ['word_mean']

    runner = TestRunner('crowdsourcing_argumentation_expts', datasets, feature_types, embeddings_types, methods,
                        dataset_increment)
    runner.run_test_set()

    datasets = ['UKPConvArgAll']
    methods = ['BI-LSTM', 'SinglePrefGP_noOpt_weaksprior']
    feature_types = ['embeddings'] # ''
    embeddings_types = ['word_mean']

    runner = TestRunner('crowdsourcing_argumentation_expts', datasets, feature_types, embeddings_types, methods,
                        dataset_increment)
    runner.run_test_set()

    datasets = ['UKPConvArgAll']
    methods = ['SVM', 'BI-LSTM', 'SinglePrefGP_noOpt_weaksprior', 'SinglePrefGP_weaksprior',
               'SingleGPC_noOpt_weaksprior', 'GP+SVM']
    feature_types = ['both']
    embeddings_types = ['word_mean']

    runner = TestRunner('crowdsourcing_argumentation_expts', datasets, feature_types, embeddings_types, methods,
                        dataset_increment)
    runner.run_test_set()