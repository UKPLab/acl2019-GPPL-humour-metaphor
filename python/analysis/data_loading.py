# -- coding: utf-8 --

'''
Helper functions for loading the data to run tests using the dataset from Ivan Habernal, 2016, ACL.

Created on 10 Jun 2017

@author: edwin
'''
import os, sys

data_root_dir = os.path.abspath(os.path.expanduser("./data/"))

sys.path.append(os.path.abspath('./git/acl2016-convincing-arguments/code/argumentation-convincingness-experiments-python'))
sys.path.append(os.path.expanduser('~/data/personalised_argumentation/embeddings/Siamese-CBOW/siamese-cbow'))
sys.path.append(os.path.expanduser("~/data/personalised_argumentation/embeddings/skip-thoughts"))

from data_loader import load_my_data_separate_args
from data_loader_regression import load_my_data as load_my_data_regression
from sklearn.datasets import load_svmlight_file
from preproc_raw_data import generate_turker_CSV, generate_gold_CSV
import numpy as np

def combine_lines_into_one_file(dataset_name, dirname=os.path.join(data_root_dir, 'lingdata/UKPConvArg1-Full-libsvm'),
                                outputfile=os.path.join(data_root_dir, 'lingdata/%s-libsvm.txt')):
    output_argid_file = outputfile % ("argids_%s" % dataset_name)
    outputfile = outputfile % dataset_name
    
    outputstr = ""
    dataids = [] # contains the argument document IDs in the same order as in the ouputfile and outputstr
    
    if os.path.isfile(outputfile):
        os.remove(outputfile)
    if os.path.isfile(output_argid_file):
        os.remove(output_argid_file)

    with open(outputfile, 'a') as ofh: 
        for filename in os.listdir(dirname):

            if os.path.samefile(outputfile, os.path.join(dirname, filename)):
                continue

            if filename.split('.')[-1] != 'txt':
                continue

            fid = filename.split('.')[0]
            print(("writing from file %s with row ID %s" % (filename, fid)))
            with open(dirname + "/" + filename) as fh:
                lines = fh.readlines()
            for line in lines:
                dataids.append(fid)
                outputline = line.split('#')[0]
                if outputline[-1] != '\n':
                    outputline += '\n'

                ofh.write(outputline)
                outputstr += outputline + '\n'

    dataids = np.array(dataids)[:, np.newaxis]
    np.savetxt(output_argid_file, dataids, '%s')
                
    return outputfile, outputstr, dataids   

def load_train_test_data(dataset):
    # Set experiment options and ensure CSV data is ready -------------------------------------------------------------
    
    folds_regression = None # test data for regression (use the folds object for training)
    folds_test = None # can load a separate set of data for testing classifications, e.g. train on workers and test on gold standard
    folds_regression_test = None # for when the test for the ranking is different to the training data
    
    # Select the directory containing original XML files with argument data + crowdsourced annotations.
    # See the readme in the data folder from Habernal 2016 for further explanation of folder names.  
    
    
    norankidx = dataset.find('_noranking')
    if norankidx > -1:
        dataset = dataset[:norankidx]
      
    if dataset == 'UKPConvArgCrowd':
        # basic dataset, requires additional steps to produce the other datasets        
        dirname = os.path.join(data_root_dir, 'argument_data/UKPConvArg1-full-XML/')
        ranking_csvdirname = os.path.join(data_root_dir, 'argument_data/UKPConvArgAllRank-CSV/')

    elif dataset == 'UKPConvArgCrowdSample':
        dirname = os.path.join(data_root_dir, 'argument_data/UKPConvArg1-crowdsample-XML/')
        ranking_csvdirname = os.path.join(data_root_dir, 'argument_data/UKPConvArg1-crowdsample-ranking-CSV/')

    elif dataset == 'UKPConvArgMACE' or dataset == 'UKPConvArgAll':
        dirname = os.path.join(data_root_dir, 'argument_data/UKPConvArg1-full-XML/')
        ranking_csvdirname = os.path.join(data_root_dir, 'argument_data/UKPConvArg1-Ranking-CSV/')

    elif dataset == 'UKPConvArgStrict':
        dirname = os.path.join(data_root_dir, 'argument_data/UKPConvArg1Strict-XML/')
        ranking_csvdirname = None        

    elif dataset == 'UKPConvArgCrowd_evalMACE': # train on the crowd dataset and evaluate on the MACE dataset
        dirname = os.path.join(data_root_dir, 'argument_data/UKPConvArg1-full-XML/')
        ranking_csvdirname = os.path.join(data_root_dir, 'argument_data/UKPConvArgAllRank-CSV/')
        folds_test, folds_regression_test, _, _, _ = load_train_test_data('UKPConvArgAll')
        dataset = 'UKPConvArgCrowd'

    elif dataset == 'UKPConvArgCrowdSample_evalMACE':
        dirname = os.path.join(data_root_dir, 'argument_data/UKPConvArg1-crowdsample-XML/')
        ranking_csvdirname = os.path.join(data_root_dir, 'argument_data/UKPConvArg1-crowdsample-ranking-CSV/')
        folds_test, folds_regression_test, _, _, _ = load_train_test_data('UKPConvArgAll')
        dataset = 'UKPConvArgCrowdSample'

    else:
        raise Exception("Invalid dataset %s" % dataset)

    if norankidx > -1:
        ranking_csvdirname = None
        folds_regression_test = None
    
    print(("Data directory = %s, dataset=%s" % (dirname, dataset)))
    csvdirname = os.path.join(data_root_dir, 'argument_data/%s-new-CSV/' % dataset)
    # Generate the CSV files from the XML files. These are easier to work with! The CSV files from Habernal do not 
    # contain all turker info that we need, so we generate them afresh here.
    if not os.path.isdir(csvdirname):
        print("Writing CSV files...")
        os.mkdir(csvdirname)
        if 'UKPConvArgCrowd' in dataset: #dataset == 'UKPConvArgCrowd': # not for CrowdSample -- why not? Should be possible.
            generate_turker_CSV(dirname, csvdirname) # select all labels provided by turkers
        else: #if 'UKPConvArgStrict' in dataset or 'UKPConvArgAll' in dataset or dataset == 'UKPConvArgCrowdSample':
            generate_gold_CSV(dirname, csvdirname) # select only the gold labels
                
    embeddings_dir = './data/'
    print(("Embeddings directory: %s" % embeddings_dir))
    
    # Load the train/test data into a folds object. -------------------------------------------------------------------
    # Here we keep each the features of each argument in a pair separate, rather than concatenating them.
    print(('Loading train/test data from %s...' % csvdirname))
    folds, word_index_to_embeddings_map, word_to_indices_map, index_to_word_map = load_my_data_separate_args(csvdirname, 
                                                                                          embeddings_dir=embeddings_dir)
    print(list(folds.keys())[0])
    print(folds[list(folds.keys())[0]]["training"][0][:20][:10])
    print(folds[list(folds.keys())[0]]["training"][1][:20][:10])
    print(folds[list(folds.keys())[0]]["training"][2][:10])
    print(folds[list(folds.keys())[0]]["training"][3][:20])
    if ranking_csvdirname is not None:             
        folds_regression, _ = load_my_data_regression(ranking_csvdirname, embeddings_dir=embeddings_dir, 
                                                      load_embeddings=True)
        
    if folds_test is not None:
        for fold in folds:
            folds[fold]["test"] = folds_test[fold]["test"]
    if folds_regression_test is not None:
        for fold in folds_regression:
            folds_regression[fold]["test"] = folds_regression_test[fold]["test"] 

    return folds, folds_regression, word_index_to_embeddings_map, word_to_indices_map, index_to_word_map
    
def load_embeddings(word_index_to_embeddings_map):
    print('Loading embeddings')
    # converting embeddings to numpy 2d array: shape = (vocabulary_size, 300)
    embeddings = np.zeros((1 + np.max(list(word_index_to_embeddings_map.keys())), len(list(word_index_to_embeddings_map.values())[0])))
    embeddings[list(word_index_to_embeddings_map.keys())] = list(word_index_to_embeddings_map.values())
    #embeddings = np.asarray([np.array(x, dtype=np.float32) for x in word_index_to_embeddings_map.values()])
    return embeddings

def load_skipthoughts_embeddings(word_to_indices_map):
    print('Loading Skip-thoughts model...')
    global skipthoughts
    import skipthoughts
    model = skipthoughts.load_model()
    return model

def load_siamese_cbow_embeddings(word_to_indices_map):
    print('Loading Siamese CBOW embeddings...')
    filename = os.path.expanduser('~/data/personalised_argumentation/embeddings/Siamese-CBOW/cosine_sharedWeights_adadelta_lr_1_noGradClip_epochs_2_batch_100_neg_2_voc_65536x300_noReg_lc_noPreInit_vocab_65535.end_of_epoch_2.pickle')
    import wordEmbeddings as siamese_cbow
    return siamese_cbow.wordEmbeddings(filename)
     
def load_ling_features(dataset,
                       root_dir=data_root_dir,
                       ling_subdir='lingdata/',
                       input_dir=os.path.join(data_root_dir, 'lingdata/UKPConvArg1-Full-libsvm'),
                       max_n_features=None):

    ling_dir = os.path.join(root_dir, ling_subdir)
    if not os.path.exists(ling_dir):
        os.mkdir(ling_dir)

    print(("Looking for linguistic features in directory %s" % ling_dir))
    print('Loading linguistic features')
    ling_file = ling_dir + "/%s-libsvm.txt" % dataset
    argids_file = ling_dir + "/%s-libsvm.txt" % ("argids_%s" % dataset)
    if not os.path.isfile(ling_file) or not os.path.isfile(argids_file):
        ling_file, _ , docids = combine_lines_into_one_file(dataset,
                                                            dirname=input_dir,
                                                            outputfile=os.path.join(ling_dir, "%s-libsvm.txt")
                                                            )
    else:
        docids = np.genfromtxt(argids_file, str)
        print('Reloaded %i docids from file. ' % len(docids))
        
    ling_feat_spmatrix, _ = load_svmlight_file(ling_file, n_features=max_n_features)
    return ling_feat_spmatrix, docids