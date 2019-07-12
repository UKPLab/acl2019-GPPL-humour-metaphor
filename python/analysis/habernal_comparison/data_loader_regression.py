from __future__ import absolute_import
from __future__ import print_function

from copy import copy

import numpy as np
from os import listdir
import vocabulary_embeddings_extractor


def load_single_file(directory, file_name, word_to_indices_map, nb_words=None):
    """
    Loads a single file and returns a tuple of x vectors and y labels
    :param directory: dir
    :param file_name: file name
    :param word_to_indices_map: words to their indices
    :param nb_words: maximum word index to be kept; otherwise treated as OOV
    :return: tuple of lists of integers
    """
    f = open(directory + file_name, 'r')
    lines = f.readlines()
    # remove first line with comments
    del lines[0]

    x_vectors = []
    y_labels = []
    id_vector = []
    turkIDs = []
    train_a1 = []
    
    for line in lines:
        # print line
        toks = line.split('\t')
        if len(toks) == 3:
            arg_id, score, a1 = toks
        elif len(toks) == 4:
            arg_id, score, a1, turkID = toks
            turkIDs.append(turkID)
        else:                
            print("Bad line in file %s: %s" % (directory + file_name, line))
            continue 
            #(arg_id, label, a1, a2)

        id_vector.append(arg_id)

        a1_tokens = vocabulary_embeddings_extractor.tokenize(a1)
        # print(a1_tokens)
        # print(a2_tokens)

        # now convert tokens to indices; set to 2 for OOV
        if word_to_indices_map is not None:
            a1_indices = [word_to_indices_map.get(word, 2) for word in a1_tokens]
        else:
            a1_indices = []

        train_a1.append(a1)

        # join them into one vector, start with 1 for start_of_sequence, add also 1 in between
        x = [1] + a1_indices + [1]
        # print(x)

        # convert score to float. The scores seem to be negated.
        y = -float(score)

        x_vectors.append(x)
        y_labels.append(y)

    # replace all word indices larger than nb_words with OOV
    if nb_words:
        x_vectors = [[2 if word_index >= nb_words else word_index for word_index in x] for x in x_vectors]

    train_instances = x_vectors
    train_labels = y_labels

    return train_instances, train_labels, id_vector, turkIDs, train_a1


def load_my_data(directory, test_split=0.2, nb_words=None, embeddings_dir='', load_embeddings=True):
    # directory = '/home/habi/research/data/convincingness/step5-gold-data/'
    # directory = '/home/user-ukp/data2/convincingness/step7-learning-11-no-eq/'
    files = listdir(directory)
    # print(files)

    for file_name in files:
        if file_name.split('.')[-1] != 'csv':                
            print("Skipping files without .csv suffix: %s" % directory + '/' + file_name)
            files.remove(file_name)

    # folds
    folds = dict()
    for file_name in files:
        training_file_names = copy(files)
        # remove current file
        training_file_names.remove(file_name)
        folds[file_name] = {"training": training_file_names, "test": file_name}

    # print(folds)
    if load_embeddings:
        word_to_indices_map, word_index_to_embeddings_map, _ = vocabulary_embeddings_extractor.load_all(
            embeddings_dir + 'vocabulary.embeddings.all.pkl.bz2')
    else:
        word_to_indices_map = None
        word_index_to_embeddings_map = None

    # results: map with fold_name (= file_name) and two tuples: (train_x, train_y), (test_x, test_y)
    output_folds_with_train_test_data = dict()

    # load all data first
    all_loaded_files = dict()
    for file_name in folds.keys():
        # print(file_name)
        test_instances, test_labels, ids, turkIDs, argtexts = load_single_file(directory, file_name, word_to_indices_map, nb_words)
        all_loaded_files[file_name] = test_instances, test_labels, ids, turkIDs, argtexts
    print("Loaded", len(all_loaded_files), "files")

    # parse each csv file in the directory
    for file_name in folds.keys():
        # print(file_name)

        # add new fold
        output_folds_with_train_test_data[file_name] = dict()

        # fill fold with train data
        current_fold = output_folds_with_train_test_data[file_name]

        test_instances, test_labels, ids, turkIDs, argtexts = all_loaded_files.get(file_name)

        # add tuple
        current_fold["test"] = test_instances, test_labels, ids, turkIDs, argtexts

        # now collect all training instances
        all_training_instances = []
        all_training_labels = []
        all_training_ids = []
        all_training_turkIDs = []
        all_training_texts = []
        for training_file_name in folds.get(file_name)["training"]:
            training_instances, training_labels, ids, turkIDs, texts = all_loaded_files.get(training_file_name)
            all_training_instances.extend(training_instances)
            all_training_labels.extend(training_labels)
            all_training_ids.extend(ids)
            all_training_turkIDs.extend(turkIDs)
            all_training_texts.extend(texts)

        current_fold["training"] = all_training_instances, all_training_labels, all_training_ids, all_training_turkIDs, \
                                    all_training_texts

    # now we should have all data loaded

    return output_folds_with_train_test_data, word_index_to_embeddings_map


def __main__():
    np.random.seed(1337)  # for reproducibility

    # todo try with 1000 and fix functionality
    max_words = 1000
    batch_size = 32
    nb_epoch = 10

    print('Loading data...')
    folds, word_index_to_embeddings_map = load_my_data("/home/user-ukp/data2/convincingness/ConvArgStrict/")

    # print statistics
    for fold in folds.keys():
        print("Fold name ", fold)
        training_instances, training_labels = folds.get(fold)["training"]
        test_instances, test_labels = folds.get(fold)["test"]

        print("Training instances ", len(training_instances), " training labels ", len(training_labels))
        print("Test instances ", len(test_instances), " test labels ", len(test_labels))

# __main__()
