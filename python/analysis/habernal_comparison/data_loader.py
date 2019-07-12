# -- coding: utf-8 --

from __future__ import absolute_import
from __future__ import print_function

import os
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
    f = open(os.path.join(directory, file_name), 'r')
    lines = f.readlines()
    # remove first line with comments
    del lines[0]

    x_vectors = []
    y_labels = []
    id_vector = []

    x_vectors_reversed = []
    y_labels_reversed = []

    for line in lines:
        # print line
        toks = line.split('\t')
        arg_id = toks[0]
        label = toks[1]
        a1 = toks[-2]
        a2 = toks[-1]
        # print(arg_id, label, a1, a2)

        id_vector.append(arg_id)

        a1_tokens = vocabulary_embeddings_extractor.tokenize(a1)
        a2_tokens = vocabulary_embeddings_extractor.tokenize(a2)

        # print(a1_tokens)
        # print(a2_tokens)

        # now convert tokens to indices; set to 2 for OOV
        a1_indices = [word_to_indices_map.get(word, 2) for word in a1_tokens]
        a2_indices = [word_to_indices_map.get(word, 2) for word in a2_tokens]

        # join them into one vector, start with 1 for start_of_sequence, add also 1 in between
        x = [1] + a1_indices + [1] + a2_indices
        # print(x)

        # let's do the oversampling trick :)
        x_reverse = [1] + a2_indices + [1] + a1_indices

        # map class to vector
        if 'a1' == label:
            y = 0
            y_reverse = 2
        elif 'a2' == label:
            y = 2
            y_reverse = 0
        else:
            y = 1
            y_reverse = 1

        x_vectors.append(x)
        y_labels.append(y)

        x_vectors_reversed.append(x_reverse)
        y_labels_reversed.append(y_reverse)

    # replace all word indices larger than nb_words with OOV
    if nb_words:
        x_vectors = [[2 if word_index >= nb_words else word_index for word_index in x] for x in x_vectors]
        x_vectors_reversed = [[2 if word_index >= nb_words else word_index for word_index in x] for x in
                              x_vectors_reversed]

    train_instances = x_vectors
    train_labels = y_labels

    return train_instances, train_labels, id_vector, x_vectors_reversed, y_labels_reversed

def load_single_file_separate_args(directory, file_name, word_to_indices_map, nb_words=None):
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

    x_vectors_a1 = []
    x_vectors_a2 = []
    train_a1 = []
    train_a2 = []
    y_labels = []
    id_vector = []
    turkerids = []
    
    for line in lines[1:]:
        # print line
        toks = line.split('\t')
        if len(toks) != 5:
            raise Exception
        arg_id, turker_id, label, a1, a2 = toks 
        # print(arg_id, label, a1, a2)

        id_vector.append(arg_id)
        turkerids.append(turker_id)

        a1_tokens = vocabulary_embeddings_extractor.tokenize(a1)
        a2_tokens = vocabulary_embeddings_extractor.tokenize(a2)

        # print(a1_tokens)
        # print(a2_tokens)

        # now convert tokens to indices; set to 2 for OOV
        a1_indices = [word_to_indices_map.get(word, 2) for word in a1_tokens]
        a2_indices = [word_to_indices_map.get(word, 2) for word in a2_tokens]
        
        train_a1.append(a1)
        train_a2.append(a2)
        
        # join them into one vector, start with 1 for start_of_sequence, add also 1 in between
        x1 = [1] + a1_indices 
        x2 = [1] + a2_indices
        # print(x)

        # map class to vector
        if 'a1' == label:
            y = 2
        elif 'a2' == label:
            y = 0
        else:
            y = 1

        x_vectors_a1.append(x1)
        x_vectors_a2.append(x2)
        y_labels.append(y)

    # replace all word indices larger than nb_words with OOV
    if nb_words:
        x_vectors_a1 = [[2 if word_index >= nb_words else word_index for word_index in x] for x in x_vectors_a1]
        x_vectors_a2 = [[2 if word_index >= nb_words else word_index for word_index in x] for x in x_vectors_a2]

    train_instances_a1 = x_vectors_a1
    train_instances_a2 = x_vectors_a2
    train_labels = y_labels

    return train_instances_a1, train_instances_a2, train_labels, id_vector, turkerids, train_a1, train_a2

def load_my_data(directory, test_split=0.2, nb_words=None, add_reversed_training_data=False, embeddings_dir=''):
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

    word_to_indices_map, word_index_to_embeddings_map = vocabulary_embeddings_extractor.load_all(embeddings_dir + 
                                                                                'vocabulary.embeddings.all.pkl.bz2')

    # results: map with fold_name (= file_name) and two tuples: (train_x, train_y), (test_x, test_y)
    output_folds_with_train_test_data = dict()

    # load all data first
    all_loaded_files = dict()
    for file_name in folds.keys():
        # print(file_name)
        test_instances, test_labels, ids, x_vectors_reversed, y_labels_reversed = load_single_file(directory, file_name,
                                                                                                   word_to_indices_map,
                                                                                                   nb_words)
        all_loaded_files[file_name] = test_instances, test_labels, ids, x_vectors_reversed, y_labels_reversed
    print("Loaded", len(all_loaded_files), "files")

    # parse each csv file in the directory
    for file_name in folds.keys():
        # print(file_name)

        # add new fold
        output_folds_with_train_test_data[file_name] = dict()

        # fill fold with train data
        current_fold = output_folds_with_train_test_data[file_name]

        test_instances, test_labels, ids, test_x_vectors_reversed, test_y_labels_reversed = all_loaded_files.get(
            file_name)

        # add tuple
        current_fold["test"] = test_instances, test_labels, ids

        # now collect all training instances
        all_training_instances = []
        all_training_labels = []
        all_training_ids = []
        for training_file_name in folds.get(file_name)["training"]:
            training_instances, training_labels, ids, x_vectors_reversed, y_labels_reversed = all_loaded_files.get(
                training_file_name)
            all_training_instances.extend(training_instances)
            all_training_labels.extend(training_labels)
            all_training_ids.extend(ids)

            if add_reversed_training_data:
                all_training_instances.extend(x_vectors_reversed)
                all_training_labels.extend(y_labels_reversed)
                all_training_ids.extend(ids)

        current_fold["training"] = all_training_instances, all_training_labels, all_training_ids

    # now we should have all data loaded

    return output_folds_with_train_test_data, word_index_to_embeddings_map

def load_my_data_separate_args(directory, test_split=0.2, nb_words=None, add_reversed_training_data=False,
                               embeddings_dir=''):
    # directory = '/home/habi/research/data/convincingness/step5-gold-data/'
    # directory = '/home/user-ukp/data2/convincingness/step7-learning-11-no-eq/'
    files = listdir(directory)
    
    for file_name in files:
        if file_name.split('.')[-1] != 'csv':                
            print("Skipping files without .csv suffix: %s" % directory + '/' + file_name)
            files.remove(file_name)
    
    # print(files)

    # folds
    folds = dict()
    for file_name in files:
        training_file_names = copy(files)
        # remove current file
        training_file_names.remove(file_name)
        folds[file_name] = {"training": training_file_names, "test": file_name}

    # print(folds)

    word_to_indices_map, word_index_to_embeddings_map, index_to_word_map = vocabulary_embeddings_extractor.load_all(
        embeddings_dir + 'vocabulary.embeddings.all.pkl.bz2')

    # results: map with fold_name (= file_name) and two tuples: (train_x, train_y), (test_x, test_y)
    output_folds_with_train_test_data = dict()

    # load all data first
    all_loaded_files = dict()
    for file_name in folds.keys():
        #print(file_name)
        test_instances_a1, test_instances_a2, test_labels, ids, turkerids, test_a1, test_a2 = \
                                    load_single_file_separate_args(directory, file_name, word_to_indices_map, nb_words)
        all_loaded_files[file_name] = test_instances_a1, test_instances_a2, test_labels, ids, turkerids, test_a1, test_a2
    print("Loaded", len(all_loaded_files), "files")

    # parse each csv file in the directory
    for file_name in folds.keys():
        #print("Test fold: ")
        #print(file_name)

        # add new fold
        output_folds_with_train_test_data[file_name] = dict()

        # fill fold with train data
        current_fold = output_folds_with_train_test_data[file_name]

        test_instances_a1, test_instances_a2, test_labels, ids, turkerids, test_a1, test_a2 = all_loaded_files.get(file_name)

        # add tuple
        current_fold["test"] = test_instances_a1, test_instances_a2, test_labels, ids, turkerids, test_a1, test_a2

        # now collect all training instances
        all_tr_instances_a1 = []
        all_tr_instances_a2 = []
        all_tr_labels = []
        all_tr_ids = []
        all_tr_turker_ids = []
        all_tr_a1 = []
        all_tr_a2 = []
        for training_file_name in folds.get(file_name)["training"]:
            tr_instances_a1, tr_instances_a2, training_labels, ids, turker_ids, tr_a1, tr_a2 = \
                                                                            all_loaded_files.get(training_file_name)
            #print("Training file: ")
            #print(training_file_name)
            all_tr_instances_a1.extend(tr_instances_a1)
            all_tr_instances_a2.extend(tr_instances_a2)
            all_tr_labels.extend(training_labels)
            all_tr_ids.extend(ids)
            all_tr_turker_ids.extend(turker_ids)
            all_tr_a1.extend(tr_a1)
            all_tr_a2.extend(tr_a2)

        current_fold["training"] = all_tr_instances_a1, all_tr_instances_a2, all_tr_labels, all_tr_ids, \
                all_tr_turker_ids, all_tr_a1, all_tr_a2

    # now we should have all data loaded

    return output_folds_with_train_test_data, word_index_to_embeddings_map, word_to_indices_map, index_to_word_map

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
