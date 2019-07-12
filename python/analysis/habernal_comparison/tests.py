# -- coding: utf-8 --

'''
Script for comparing our Bayesian preference learning approach with the results from Habernal 2016. 

Steps in this test:

1. Load word embeddings for the original text data that were used in the NN approach in Habernal 2016. -- done, but 
only using averages to combine them.
2. Load feature data that was used in the SVM-based approach in Habernal 2016.
3. Load the crowdsourced data. -- done. 
4. Copy a similar testing setup to Habernal 2016 (training/test split?) and run the Bayesian approach (during testing,
we can set aside some held-out data). -- done, results saved to file with no metrics computed yet except acc. 
5. Print some simple metrics that are comparable to those used in Habernal 2016. 


Thoughts:
1. NN takes into account sequence of word embeddings; here we need to use a combined embedding for whole text to avoid
a 300x300 dimensional input space.
2. So our method can only learn which elements of the embedding are important, but cannot learn from patterns in the 
sequence, unless we can find a way to encode those.
3. However, the SVM-based approach also did well. Which method is better, NN or SVM, and by how much? 
4. We should be able to improve on the SVM-based approach.
5. The advantages of our method: ranking with sparse data; personalised predictions to the individual annotators; 
uncertainty estimates for active learning and decision-making confidence thresholds. 

Created on 20 Mar 2017

@author: simpson
'''

import logging
from scipy.stats.stats import pearsonr
from sklearn.svm.classes import NuSVR, SVC
logging.basicConfig(level=logging.DEBUG)

import sys
import os
from sklearn.metrics.ranking import roc_auc_score

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
import time

from gp_pref_learning import GPPrefLearning, pref_likelihood
from gp_classifier_svi import GPClassifierSVI
from gp_classifier_vb import compute_median_lengthscales
from sklearn.svm import SVR 
from data_loading import load_train_test_data, load_embeddings, load_ling_features, data_root_dir, \
    load_siamese_cbow_embeddings, load_skipthoughts_embeddings
import numpy as np
    
ndebug_features = 10
verbose = False
    
def save_fold_order(resultsdir, folds=None, dataset=None):
    if folds is None and dataset is not None:
        folds, _, _, _, _ = load_train_test_data(dataset)        
    elif folds is None:
        print("Need to provide a dataset label or a set of fold data...")
        return 
    np.savetxt(resultsdir + "/foldorder.txt", np.array(list(folds.keys()))[:, None], fmt="%s")
    
# Lengthscale initialisation -------------------------------------------------------------------------------------------
# use the median heuristic to find a reasonable initial length-scale. This is the median of the distances.
# First, grab a sample of points because N^2 could be too large.    
def compute_lengthscale_heuristic(feature_type, embeddings_type, embeddings, ling_feat_spmatrix, docids, folds, 
                                  index_to_word_map, multiply_heuristic_power=1.0):
    # get the embedding values for the test data -- need to find embeddings of the whole piece of text
    if feature_type == 'both' or feature_type == 'embeddings' or feature_type == 'debug':
        
        docidxs = []
        doc_tok_seqs = []
        doctexts = []
        for f in folds:
            doc_tok_seqs.append(folds.get(f)["test"][0])
            doc_tok_seqs.append(folds.get(f)["test"][1])
        
            testids = np.array([ids_pair.split('_') for ids_pair in folds.get(f)["test"][3]])
            docidxs.append(get_docidxs_from_ids(docids, testids[:, 0]))
            docidxs.append(get_docidxs_from_ids(docids, testids[:, 1]))
            
            doctexts.append(folds.get(f)["test"][5])
            doctexts.append(folds.get(f)["test"][6])
        
        X, _, utexts = get_doc_token_seqs(docidxs, doc_tok_seqs, doctexts)
        
        if embeddings_type == 'word_mean':
            items_feat = get_mean_embeddings(embeddings, X)
        elif embeddings_type == 'skipthoughts':
            global skipthoughts
            import skipthoughts        
            items_feat = skipthoughts.encode(embeddings, utexts)
        elif embeddings_type == 'siamese-cbow':
            items_feat = np.array([embeddings.getAggregate(index_to_word_map[Xi]) for Xi in X])
        else:
            logging.info("invalid embeddings type! %s" % embeddings_type)
        
    if feature_type == 'both' or feature_type == 'debug':
        items_feat = np.concatenate((items_feat, ling_feat_spmatrix.toarray()), axis=1)
        
    if feature_type == 'ling':
        items_feat = ling_feat_spmatrix.toarray()
    
    if feature_type == 'debug':
        items_feat = items_feat[:, :ndebug_features]
    
    starttime = time.time()
                                
    #for f in range(items_feat.shape[1]):  
    ls_initial = compute_median_lengthscales(items_feat, multiply_heuristic_power, N_max=3000)
            
    endtime = time.time()
    logging.info('@@@ Selected initial lengthscales in %f seconds' % (endtime - starttime))
    
    return ls_initial     
    
def get_doc_token_seqs(ids, X_list, texts=None):
    '''
    ids -- list of document IDs
    X_list -- list of lists of word indices for each argument corresponding to the ids
    texts -- list of texts corresponding to the ids
    returns
    X -- list of lists of word indices for each argument corresponding to the uids
    uids -- list of unique document IDs
    utexts -- unique texts corresponding to the uids
    '''
    # X_train_a1 and a1_train both have one entry per observation. We want to replace them with a list of 
    # unique arguments, and the indexes into that list. First, get the unique argument ids from trainids and testids:
    if hasattr(ids[0], '__len__'):
        allids = np.concatenate(ids)
    else:
        allids = ids
    uids, uidxs = np.unique(allids, return_index=True)
    # get the word index vectors corresponding to the unique arguments
    X = np.empty(np.max(uids) + 1, dtype=object)

    if texts is not None:
        utexts = np.zeros(np.max(uids) + 1, dtype=object)
        utexts[:] = '' 
    
    start = 0
    fin = 0

    for i in range(len(X)):
        if i not in uids:
            X[i] = []

    for i in range(len(X_list)):
        fin += len(X_list[i])
        idxs = (uidxs>=start) & (uidxs<fin)
        # keep the original IDs to try to make life easier. This means the IDs become indexes into X    
        X[uids[idxs]] = np.array(X_list[i])[uidxs[idxs] - start]
        
        if texts is not None:
            utexts[uids[idxs]] = np.array(texts[i])[uidxs[idxs] - start] 
        
        start += len(X_list[i])
        
    if texts is not None:
        utexts = [utext for utext in utexts]
        return X, uids, utexts
    else:
        return X, uids    
    
def get_mean_embeddings(word_embeddings, X):
    return np.array([np.mean(word_embeddings[Xi, :], axis=0) for Xi in X])    

def get_docidxs_from_ids(all_docids, ids_to_map):
    return np.array([np.argwhere(docid==all_docids)[0][0] for docid in ids_to_map])
    
def get_fold_data(folds, fold, docids):
    #X_train_a1, X_train_a2 are lists of lists of word indexes 
    X_train_a1, X_train_a2, prefs_train, ids_train, person_train, tr_a1, tr_a2 = folds.get(fold)["training"]
    X_test_a1, X_test_a2, prefs_test, ids_test, person_test, test_a1, test_a2 = folds.get(fold)["test"]
    
    #a1_train, a2_train are lists of argument ids
    trainids = np.array([ids_pair.split('_') for ids_pair in ids_train])
    if docids is None:
        docids = np.arange(np.unique(trainids).size)
    a1_train = get_docidxs_from_ids(docids, trainids[:, 0])
    a2_train = get_docidxs_from_ids(docids, trainids[:, 1])
    
    testids = np.array([ids_pair.split('_') for ids_pair in ids_test])
    a1_test = get_docidxs_from_ids(docids, testids[:, 0])
    a2_test = get_docidxs_from_ids(docids, testids[:, 1])

    X, uids, utexts = get_doc_token_seqs((a1_train, a2_train, a1_test, a2_test), 
                           [X_train_a1, X_train_a2, X_test_a1, X_test_a2], (tr_a1, tr_a2, test_a1, test_a2))
        
    print(("Training instances ", len(X_train_a1), " training labels ", len(prefs_train)))
    print(("Test instances ", len(X_test_a1), " test labels ", len(prefs_test)))
    
    prefs_train = np.array(prefs_train) 
    prefs_test = np.array(prefs_test)     
    person_train = np.array(person_train)
    person_test = np.array(person_test)  
    
    personIDs = np.concatenate((person_train, person_test))
    _, personIdxs = np.unique(personIDs, return_inverse=True)
    person_train = personIdxs[:len(person_train)]
    person_test = personIdxs[len(person_train):]
    
    return a1_train, a2_train, prefs_train, person_train, a1_test, a2_test, prefs_test, person_test, \
        X, uids, utexts
        
def get_noisy_fold_data(folds, fold, docids, acc, tr_pair_subset=None):
    a1_train, a2_train, prefs_train, person_train, a1_test, a2_test, prefs_test, person_test, X, \
    uids, utexts = get_fold_data(folds, fold, docids)
    
    # now subsample the training data
    N = len(a1_train)
    if tr_pair_subset is not None:
        Nsub = N * tr_pair_subset
        subidxs = np.random.choice(N, Nsub, replace=False)
        a1_train = a1_train[subidxs]
        a2_train = a2_train[subidxs]
        prefs_train = prefs_train[subidxs]
        person_train = person_train[subidxs]
    else:
        Nsub = N

    if acc != 1.0:
        # now we add noise to the training data
        flip_labels = np.random.rand(Nsub) > acc
        prefs_train[flip_labels] = 2 - prefs_train[flip_labels] # labels are 0, 1 or 2
    
    return a1_train, a2_train, prefs_train, person_train, a1_test, a2_test, prefs_test, person_test, \
        X, uids, utexts   
    
def get_fold_regression_data(folds_regression, fold, docids):
    if folds_regression is not None:
        _, scores_rank_train, argids_rank_train, person_rank_train, _ = folds_regression.get(fold)["training"] # blank argument is turkIDs_rank_test
        item_idx_ranktrain = np.array([np.argwhere(trainid==docids)[0][0] for trainid in argids_rank_train])
        scores_rank_train = np.array(scores_rank_train)
        argids_rank_train = np.array(argids_rank_train)    
        
        _, scores_rank_test, argids_rank_test, personIDs_rank_test, _ = folds_regression.get(fold)["test"] # blank argument is turkIDs_rank_test
        item_idx_ranktest = np.array([np.argwhere(testid==docids)[0][0] for testid in argids_rank_test])
        scores_rank_test = np.array(scores_rank_test)
        argids_rank_test = np.array(argids_rank_test)
    else:
        item_idx_ranktrain = None
        scores_rank_train = None
        argids_rank_train = None
        person_rank_train = None
        
        item_idx_ranktest = None
        scores_rank_test = None
        argids_rank_test = None
        personIDs_rank_test = None    

    return item_idx_ranktrain, scores_rank_train, argids_rank_train, person_rank_train,\
           item_idx_ranktest, scores_rank_test, argids_rank_test, personIDs_rank_test
    
    
def subsample_tr_data(subsample_amount, a1_train, a2_train):
            
    item_subsample_ids = []
    nselected = 0
    while nselected < subsample_amount:
        idx = np.random.choice(len(a1_train), 1)
        
        if a1_train[idx] not in item_subsample_ids:
            item_subsample_ids.append(a1_train[idx])
    
        if a2_train[idx] not in item_subsample_ids:
            item_subsample_ids.append(a2_train[idx])

        nselected = len(item_subsample_ids)
    
    pair_subsample_idxs = np.argwhere(np.in1d(a1_train, item_subsample_ids) & np.in1d(a2_train, item_subsample_ids)).flatten()
    
    #    pair_subsample_idxs = np.random.choice(len(a1_train), subsample_amount, replace=False)
    
    return pair_subsample_idxs
    
class TestRunner:    
    
    def __init__(self, current_expt_output_dir, datasets, feature_types, embeddings_types, methods, 
                 dataset_increment, expt_tag='habernal'):    
        self.folds = None
        self.initial_pair_subset = {}
        self.default_ls_values = {}
        
        self.expt_output_dir = current_expt_output_dir
        self.expt_tag = expt_tag
        
        self.datasets = datasets
        self.feature_types = feature_types
        self.embeddings_types = embeddings_types
        self.methods = methods
        self.dataset_increment = dataset_increment
        
    def load_features(self, feature_type, embeddings_type, a1_train, a2_train, uids, utexts=None):
        '''
        Load all the features specified by the type into an items_feat object. Remove any features where the values are all
        zeroes.
        '''
        # get the embedding values for the test data -- need to find embeddings of the whole piece of text
        if feature_type == 'both' or feature_type == 'embeddings' or feature_type=='debug':
            logging.info("Converting texts to mean embeddings (we could use a better sentence embedding?)...")
            if embeddings_type == 'word_mean':
                items_feat = get_mean_embeddings(self.embeddings, self.X)
            elif embeddings_type == 'skipthoughts':
                global skipthoughts
                import skipthoughts
                items_feat = skipthoughts.encode(self.embeddings, utexts)
            elif embeddings_type == 'siamese-cbow':
                items_feat = np.array([self.embeddings.getAggregate(self.index_to_word_map[Xi]) for Xi in self.X])
            else:
                logging.info("invalid embeddings type! %s" % embeddings_type)
            logging.info("...embeddings loaded.")
            # trim away any features not in the training data because we can't learn from them
            valid_feats = np.sum((items_feat[a1_train] != 0) + (items_feat[a2_train] != 0), axis=0) > 0
            items_feat = items_feat[:, valid_feats]
            self.ling_items_feat = None # will get overwritten if we load the linguistic features further down.
            self.embeddings_items_feat = items_feat
            
        elif feature_type == 'ling':
            items_feat = np.zeros((self.X.shape[0], 0))
            valid_feats = np.zeros(0)
            self.embeddings_items_feat = None
            
        if feature_type == 'both' or feature_type == 'ling' or feature_type == 'debug':
            logging.info("Obtaining linguistic features for argument texts.")
            # trim the features that are not used in training
            valid_feats_ling = np.sum( (self.ling_feat_spmatrix[a1_train, :] != 0) + 
                                       (self.ling_feat_spmatrix[a2_train, :] != 0), axis=0) > 0 
            valid_feats_ling = np.array(valid_feats_ling).flatten()
            self.ling_items_feat = self.ling_feat_spmatrix[uids, :][:, valid_feats_ling].toarray()
            items_feat = np.concatenate((items_feat, self.ling_items_feat), axis=1)
            logging.info("...loaded all linguistic features for training and test data.")
            valid_feats = np.concatenate((valid_feats, valid_feats_ling))
            
        print('Found %i features.' % items_feat.shape[1])
            
        if feature_type=='debug':
            items_feat = items_feat[:, :ndebug_features] #use only n features for faster debugging
            self.ling_items_feat = items_feat
            self.embeddings_items_feat = items_feat
            valid_feats = valid_feats[:ndebug_features]
            
        self.items_feat = items_feat
        self.ndims = self.items_feat.shape[1]
        self.valid_feats = valid_feats.astype(bool)
    
    # Methods for running the prediction methods --------------------------------------------------------------------------
    def run_gppl(self):
        # TODO: Find out whether updates to preference learning code or the test framework seem to have reduced accuracy.
        #   - Convergence taking longer, method runs for 200 iterations without completing. Maybe step size in the SVI
        # updates should be increased, but this does not explain the change. ***Caused by change to logpt?*** 
        # ***Covariance means we can't treat f_var in same way as noise! Was there some reason we previously thought the var cancelled out?***
        #   - Delay changed from 1 to 10. This may mean it doesn't converge in the permitted no. iterations. ***Testing again with delay of 1 didn't change result much***
        #   - Lengthscale median heuristic should be the same. 
        #   - Changes to the way the output is computed to account for uncertainty in f might be responsible? This 
        # could be making data points with higher mean phi but greater uncertainty have same predictions as those with 
        # lower mean phi and greater certainty. 
        
        if 'additive' in self.method:
            kernel_combination = '+'
        else:
            kernel_combination = '*'
            
        if 'shrunk' in self.method:
            ls_initial = self.ls_initial / float(len(self.ls_initial))
        else:
            ls_initial = self.ls_initial
        
        if 'weaksprior' in self.method:
            shape_s0 = 2.0
            rate_s0 = 200.0
        elif 'lowsprior' in self.method:
            shape_s0 = 1.0
            rate_s0 = 1.0
        elif 'weakersprior' in self.method:
            shape_s0 = 2.0
            rate_s0 = 2000.0
        else:
            shape_s0 = 200.0
            rate_s0 = 20000.0
            
        if '_M' in self.method:
            validx = self.method.find('_M') + 2
            M = int(self.method[validx:])
        else:
            M = 500
            
        if '_SS' in self.method:
            validx = self.method.find('_SS') + 3
            SS = int(self.method[validx:])
        else:
            SS = 200        
        
        if self.model is None:
            
            if M == 0:
                use_svi = False
            else:
                use_svi = True
            
            self.model = GPPrefLearning(ninput_features=self.ndims, ls_initial=ls_initial, verbose=self.verbose, 
                    shape_s0=shape_s0, rate_s0=rate_s0, rate_ls = 1.0 / np.mean(ls_initial), use_svi=use_svi, 
                    ninducing=M, max_update_size=SS, kernel_combination=kernel_combination, forgetting_rate=0.7, 
                    delay=1.0)
            self.model.max_iter_VB = 2000
            new_items_feat = self.items_feat # pass only when initialising
        else:
            new_items_feat = None
        
        print("no. features: %i" % new_items_feat.shape[1])
        self.model.fit(self.a1_train, self.a2_train, new_items_feat, np.array(self.prefs_train, dtype=float)-1, 
                  optimize=self.optimize_hyper, input_type='zero-centered')            
    
        proba = self.model.predict(None, self.a1_test, self.a2_test, reuse_output_kernel=True, return_var=False)
    
        if self.a1_unseen is not None and len(self.a1_unseen):
            tr_proba, _ = self.model.predict(None, self.a1_unseen, self.a2_unseen, reuse_output_kernel=True)
        else:
            tr_proba = None
        
        if self.a_rank_test is not None:
            predicted_f, _ = self.model.predict_f(None, self.a_rank_test)
        else:
            predicted_f = None

        return proba, predicted_f, tr_proba
    
#     model, _, a1_train, a2_train, self.prefs_train, items_feat, _, _, self.a1_test, self.a2_test, 
#                   self.a1_unseen, self.a2_unseen, ls_initial, verbose, _, self.a_rank_test=None, _, _, _
    def run_gpsvm(self):
        if self.model is None:
            self.model = GPPrefLearning(ninput_features=1, ls_initial=[1000], verbose=self.verbose, shape_s0 = 1.0, 
                        rate_s0 = 1.0, rate_ls = 1.0 / np.mean(self.ls_initial), use_svi=False, kernel_func='diagonal')
            self.model.max_iter_VB = 10
            
        # never use optimize with diagonal kernel
        self.model.fit(self.a1_train, self.a2_train, np.arange(self.items_feat.shape[0])[:, np.newaxis], 
                  np.array(self.prefs_train, dtype=float)-1, optimize=False, input_type='zero-centered')             
    
        train_idxs = np.unique([self.a1_train, self.a2_train])
        train_feats = self.items_feat[train_idxs]
        f, _ = self.model.predict_f(train_idxs[:, np.newaxis])
        svm = SVR()
        svm.fit(train_feats, f)
        test_f = svm.predict(self.items_feat)
        
        # apply the preference likelihood from GP method
        proba = pref_likelihood(test_f, v=self.a1_test, u=self.a2_test, return_g_f=False)
        if self.a_rank_test is not None:
            predicted_f = svm.predict(self.items_feat[self.a_rank_test])
        else:
            predicted_f = None  
    
        if self.a1_unseen is not None and len(self.a1_unseen):
            tr_proba = pref_likelihood(test_f, v=self.a1_unseen, u=self.a2_unseen, return_g_f=False)
        else:
            tr_proba = None
            
        return proba, predicted_f, tr_proba      
        
    def run_gpc(self):
        if 'additive' in self.method:
            kernel_combination = '+'
        else:
            kernel_combination = '*'
        
        if 'weaksprior' in self.method:
            shape_s0 = 2.0
            rate_s0 = 200.0
        elif 'lowsprior':
            shape_s0 = 1.0
            rate_s0 = 1.0
        else:
            shape_s0 = 200.0
            rate_s0 = 20000.0
    
        # twice as many features means the lengthscale heuristic is * 2
        if self.model is None:
            ls_initial = np.concatenate((self.ls_initial * 2.0, self.ls_initial * 2.0))
            self.model = GPClassifierSVI(ninput_features=self.ndims, ls_initial=ls_initial, verbose=self.verbose, 
                                shape_s0=shape_s0, rate_s0=rate_s0, rate_ls=1.0 / np.mean(self.ls_initial), 
                                use_svi=True, ninducing=500, max_update_size=200, kernel_combination=kernel_combination)            
            self.model.max_iter_VB = 500
              
        # with the argument order swapped around and data replicated:
        gpc_feats = np.empty(shape=(len(self.a1_train)*2, self.items_feat.shape[1]*2))
        gpc_feats[:len(self.a1_train), :self.items_feat.shape[1]] = self.items_feat[self.a1_train]
        gpc_feats[len(self.a1_train):, :self.items_feat.shape[1]] = self.items_feat[self.a2_train]
        gpc_feats[:len(self.a1_train), self.items_feat.shape[1]:] = self.items_feat[self.a2_train]
        gpc_feats[len(self.a1_train):, self.items_feat.shape[1]:] = self.items_feat[self.a1_train]
        
        gpc_labels = np.concatenate((np.array(self.prefs_train, dtype=float) * 0.5,
                                      1 - np.array(self.prefs_train, dtype=float) * 0.5))
                  
        self.model.fit(np.arange(len(self.a1_train)), gpc_labels, optimize=self.optimize_hyper, features=gpc_feats)            
        
        proba, _ = self.model.predict(np.concatenate((self.items_feat[self.a1_test], self.items_feat[self.a2_test]), axis=1))
        if self.a_rank_test is not None:
            predicted_f = np.zeros(len(self.a_rank_test)) # can't easily rank with this method
        else:
            predicted_f = None
    
        if self.a1_unseen is not None and len(self.a1_unseen):
            tr_proba, _ = self.model.predict(np.concatenate((self.items_feat[self.a1_unseen], self.items_feat[self.a2_unseen]), axis=1))
        else:
            tr_proba = None
    
        return proba, predicted_f, tr_proba      
    
    def run_svm(self, feature_type):
#         from svmutil import svm_train, svm_predict, svm_read_problem
#          
#         if feature_type == 'embeddings' or feature_type == 'both' or feature_type == 'debug':
#             embeddings = self.embeddings_items_feat
#         else:
#             embeddings = None
         
        prefs_train_fl = np.array(self.prefs_train, dtype=float)
        svc_labels = np.concatenate((prefs_train_fl * 0.5, 1 - prefs_train_fl * 0.5))
#                                             
#         filetemplate = os.path.join(data_root_dir, 'libsvmdata/%s-%s-%s-libsvm.txt')
#         nfeats = self.ling_feat_spmatrix.shape[1]
#           
#         #if not os.path.isfile(trainfile):
#         trainfile, _, _ = combine_into_libsvm_files(self.dataset, self.docids[self.a1_train], self.docids[self.a2_train], 
#             svc_labels, 'training', self.fold, nfeats, outputfile=filetemplate, reverse_pairs=True, embeddings=embeddings, 
#             a1=self.a1_train, a2=self.a2_train, embeddings_only=feature_type=='embeddings')
#           
#         problem = svm_read_problem(trainfile) 
#         self.model = svm_train(problem[0], problem[1], '-b 1')
#       
#         #if not os.path.isfile(testfile):
#         testfile, _, _ = combine_into_libsvm_files(self.dataset, self.docids[self.a1_test], self.docids[self.a2_test], 
#             np.ones(len(self.a1_test)), 'test', self.fold, nfeats, outputfile=filetemplate, embeddings=embeddings, 
#             a1=self.a1_test, a2=self.a2_test, embeddings_only=feature_type=='embeddings')
#              
#         problem = svm_read_problem(testfile)        
#         _, _, proba = svm_predict(problem[0], problem[1], self.model, '-b 1')

        svc = SVC(probability=True)
        trainfeats = np.concatenate((np.concatenate((self.items_feat[self.a1_train], self.items_feat[self.a2_train]), axis=1),
                               np.concatenate((self.items_feat[self.a1_train], self.items_feat[self.a2_train]), axis=1)),
                               axis=0)
        print("no. features: %i" % trainfeats.shape[1])
        print("no. pairs: %i" % trainfeats.shape[0])
        svc.fit(trainfeats, svc_labels)
        proba = svc.predict_proba(np.concatenate((self.items_feat[self.a1_train], self.items_feat[self.a2_train]), axis=1))

        # libSVM flips the labels if the first one it sees is positive
        if svc_labels[0] == 1:
            proba = 1 - np.array(proba)
         
        if self.a_rank_test is not None:
            svr = NuSVR()    
            svr.fit(self.items_feat[self.a_rank_train], self.scores_rank_train)
            predicted_f = svr.predict(self.items_feat[self.a_rank_test])
            logging.debug('Predictions from SVM regression: %s ' % predicted_f)
        else:
            predicted_f = None
    
        if self.a1_unseen is not None and len(self.a1_unseen):
#             testfile, _, _ = combine_into_libsvm_files(self.dataset, self.docids[self.a1_unseen], 
#                                                        self.docids[self.a2_unseen], np.ones(len(self.a1_unseen)), 
#                                            'unseen', self.fold, nfeats, outputfile=filetemplate, embeddings=embeddings,
#                                            a1=self.a1_unseen, a2=self.a2_unseen, embeddings_only=feature_type=='embeddings')
#             
#             problem = svm_read_problem(testfile)
#             _, _, tr_proba = svm_predict(problem[0], problem[1], self.model, '-b 1')

            tr_proba = svc.predict_proba(np.concatenate((self.items_feat[self.a1_unseen], self.items_feat[self.a2_unseen]), axis=1))

            # libSVM flips the labels if the first one it sees is positive
            if svc_labels[0] == 1:
                tr_proba = 1 - np.array(tr_proba)
        else:
            tr_proba = None
    
        return proba[:, None], predicted_f, tr_proba[:, None] 
     
    def run_bilstm(self, feature_type):     
        from keras.preprocessing import sequence
        from keras.models import Graph
        from keras.layers.core import Dense, Dropout
        from keras.layers.embeddings import Embedding
        from keras.layers.recurrent import LSTM
            
        # Include document-level features in a simple manner using one hidden layer, which is then combined with the outputs of
        # the LSTM layers, as in "Boosting Information Extraction Systems with Character-level Neural Networks and Free Noisy 
        # Supervision". This is equivalent to an MLP with one hidden layer combined with the LSTM.
        if feature_type == 'ling' or feature_type == 'both' or feature_type == 'debug':
            use_doc_level_features = True
            n_doc_level_feats = self.ling_items_feat.shape[1] 
        else:
            use_doc_level_features = False
            
        np.random.seed(1337) # for reproducibility         
        max_len = 300  # cut texts after this number of words (among top max_features most common words)
        batch_size = 32
        nb_epoch = 5  # 5 epochs are meaningful to prevent over-fitting...
     
        print(len(self.folds.get(self.fold)["training"]))
        X_train1 = self.X[self.a1_train]
        X_train2 = self.X[self.a2_train]
        y_train = self.prefs_train.tolist()
        X_train = []
        for i, row1 in enumerate(X_train1):
            row1 = row1 + X_train2[i]
            X_train.append(row1)
        X_test1, X_test2, _, _, _, _, _ = self.folds.get(self.fold)["test"]
        X_test = []
        for i, row1 in enumerate(X_test1):
            row1 = row1 + X_test2[i]
            X_test.append(row1)        
        print("Pad sequences (samples x time)")
        X_train = sequence.pad_sequences(X_train, maxlen=max_len)
        X_test = sequence.pad_sequences(X_test, maxlen=max_len)
        print(('X_train shape:', X_train.shape))
        print(('X_test shape:', X_test.shape))
        y_train = np.array(y_train) / 2.0
        print(('y_train values: ', np.unique(y_train)))
     
        print('Training data sizes:')
        print((X_train.shape))
        print((y_train.shape))
        if use_doc_level_features:
            pair_doc_feats_tr = np.concatenate((self.ling_items_feat[self.a1_train, :], 
                                                  self.ling_items_feat[self.a2_train, :]), axis=1)         
            print((pair_doc_feats_tr.shape))
            print(n_doc_level_feats * 2)
            
            pair_doc_feats_test = np.concatenate((self.ling_items_feat[self.a1_test, :], 
                                                  self.ling_items_feat[self.a2_test, :]), axis=1)
     
        print('Build model...')
        if self.model is None:
            self.model = Graph()
            self.model.add_input(name='input', input_shape=(max_len,), dtype='int')
            self.model.add_node(Embedding(self.embeddings.shape[0], self.embeddings.shape[1], input_length=max_len, 
                                     weights=[self.embeddings]), name='embedding', input='input')
            self.model.add_node(LSTM(64), name='forward', input='embedding')
            self.model.add_node(LSTM(64, go_backwards=True), name='backward', input='embedding')
            self.model.add_node(Dropout(0.5), name='dropout', inputs=['forward', 'backward'])
            
            if use_doc_level_features:
                self.model.add_input(name='docfeatures', input_shape=(n_doc_level_feats*2,), dtype='float')
                self.model.add_node(Dense(64, activation='relu'), name='docfeatures_hiddenlayer', input='docfeatures')
                self.model.add_node(Dropout(0.5), name='dropout_docfeatures', input='docfeatures_hiddenlayer')
                
                self.model.add_node(Dense(1, activation='sigmoid'), name='sigmoid',
                                    inputs=['dropout_docfeatures', 'dropout'])
            else:
                self.model.add_node(Dense(1, activation='sigmoid'), name='sigmoid', input='dropout')            
            
            self.model.add_output(name='output', input='sigmoid')
          
            # try using different optimizers and different optimizer configs
            self.model.compile('adam', {'output': 'binary_crossentropy'})
          
        print('Train...')
        if use_doc_level_features:
            self.model.fit({'input': X_train, 'docfeatures' : pair_doc_feats_tr, 'output': y_train}, 
                       batch_size=batch_size, nb_epoch=nb_epoch)
        else:
            self.model.fit({'input': X_train, 'output': y_train}, batch_size=batch_size, nb_epoch=nb_epoch)
      
        print('Prediction')
        if use_doc_level_features:
            model_predict = self.model.predict({'input': X_test, 'docfeatures': pair_doc_feats_test}, batch_size=batch_size)
        else:
            model_predict = self.model.predict({'input': X_test}, batch_size=batch_size)
        proba = np.array(model_predict['output'])
         
        #proba = np.zeros(len(prefs_test))
        if self.a_rank_test is not None:
            X_train = self.X[self.a_rank_train]
            X_test = self.X[self.a_rank_test]
         
            print((len(X_train), 'train sequences'))
            print((len(X_test), 'test sequences'))
         
            print("Pad sequences (samples x time)")
            X_train = sequence.pad_sequences(X_train, maxlen=max_len)
            X_test = sequence.pad_sequences(X_test, maxlen=max_len)
            print(('X_train shape:', X_train.shape))
            print(('X_test shape:', X_test.shape))
         
            print('Build model...')
            rank_model = Graph()
            rank_model.add_input(name='input', input_shape=(max_len,), dtype=int)
            rank_model.add_node(Embedding(self.embeddings.shape[0], self.embeddings.shape[1], input_length=max_len, 
                                          weights=[self.embeddings]), name='embedding', input='input')
            rank_model.add_node(LSTM(64), name='forward', input='embedding')
            rank_model.add_node(LSTM(64, go_backwards=True), name='backward', input='embedding')
            rank_model.add_node(Dropout(0.5), name='dropout', inputs=['forward', 'backward'])
         
            # match output layer for regression better
            if use_doc_level_features:
                rank_model.add_input(name='docfeatures', input_shape=(n_doc_level_feats,), dtype='float')
                rank_model.add_node(Dense(64, activation='relu'), name='docfeatures_hiddenlayer', input='docfeatures')
                rank_model.add_node(Dropout(0.5), name='dropout_docfeatures', input='docfeatures_hiddenlayer')
                
                rank_model.add_node(Dense(1, activation='linear', init='uniform'), name='output_layer', 
                                    inputs=['dropout_docfeatures', 'dropout'])
            else:            
                rank_model.add_node(Dense(1, activation='linear', init='uniform'), name='output_layer', input='dropout')
            rank_model.add_output(name='output', input='output_layer')
         
            # use mean absolute error loss
            rank_model.compile('adam', {'output': 'mean_absolute_error'})
         
            print('Train...')
            if use_doc_level_features:
                rank_model.fit({'input': X_train, 'docfeatures' : self.ling_items_feat[self.a_rank_train, :], 'output': 
                            self.scores_rank_train}, batch_size=batch_size, nb_epoch=nb_epoch)
            else:
                rank_model.fit({'input': X_train, 'output': self.scores_rank_train}, batch_size=batch_size, nb_epoch=nb_epoch)
                
            print('Prediction')
            if use_doc_level_features:
                model_predict = rank_model.predict({'input': X_test, 'docfeatures': self.ling_items_feat[self.a_rank_test, :]}, 
                                               batch_size=batch_size)
            else:
                model_predict = rank_model.predict({'input': X_test}, batch_size=batch_size)
            predicted_f = np.asarray(model_predict['output']).flatten()                
     
            print(('Unique regression predictions: ', np.unique(predicted_f)))
        else:
            predicted_f = None
    
        if self.a1_unseen is not None and len(self.a1_unseen):
            X_test = []
            X_test1 = self.X[self.a1_unseen]
            X_test2 = self.X[self.a2_unseen]
            for i, row1 in enumerate(X_test1):
                row1 = row1 + X_test2[i]
                X_test.append(row1)   
            X_test = sequence.pad_sequences(X_test, maxlen=max_len)
            print('Prediction on unseen data...')
            if use_doc_level_features:
                pair_doc_feats_unseen = np.concatenate((self.ling_items_feat[self.a1_unseen, :], 
                                                        self.ling_items_feat[self.a2_unseen, :]), axis=1) 
                model_predict = self.model.predict({'input': X_test, 'docfeatures': pair_doc_feats_unseen}, batch_size=batch_size)
            else:
                model_predict = self.model.predict({'input': X_test}, batch_size=batch_size)
            tr_proba = np.array(model_predict['output'])
        else:
            tr_proba = None
    
        
        return proba, predicted_f, tr_proba           
       
    def _choose_method_fun(self, feature_type):
        if 'SinglePrefGP' in self.method:
            method_runner_fun = self.run_gppl
        elif 'GP+SVM' in self.method:
            method_runner_fun = self.run_gpsvm
        elif 'SingleGPC' in self.method:
            method_runner_fun = self.run_gpc
        elif 'SVM' in self.method:
            method_runner_fun = lambda: self.run_svm(feature_type)
        elif 'BI-LSTM' in self.method:
            if feature_type == 'ling':
                logging.error("BI-LSTM is not set up to run without using embeddings. Will switch to feature type=both...")
                feature_type = 'both'            
            method_runner_fun = lambda: self.run_bilstm(feature_type)    
            
        return method_runner_fun
               
    def _set_resultsfile(self, feature_type, embeddings_type, acc, dataset_increment):
        # To run the active learning tests, call this function with dataset_increment << 1.0. 
        # To add artificial noise to the data, run with acc < 1.0.
        output_data_dir = os.path.join(data_root_dir, 'outputdata/')
        if not os.path.isdir(output_data_dir):
            os.mkdir(output_data_dir)
            
        output_data_dir = os.path.join(output_data_dir, self.expt_output_dir)
        if not os.path.isdir(output_data_dir):
            os.mkdir(output_data_dir)    
            
        # Select output paths for CSV files and final results
        output_filename_template = output_data_dir + '/%s' % self.expt_tag
        output_filename_template += '_%s_%s_%s_%s_acc%.2f_di%.2f' 
    
        results_stem = output_filename_template % (self.dataset, self.method, feature_type, embeddings_type, acc, 
                                                                  dataset_increment) 
        resultsfile = results_stem + '_test.pkl' # the old results format with everything in one file
    #     modelfile = results_stem + '_model_fold%i.pkl'
                    
        logging.info('The output file for the results will be: %s' % resultsfile)    
        
        if not os.path.isdir(results_stem):
            os.mkdir(results_stem)
            
        return resultsfile, results_stem    
    
    def _load_dataset(self, dataset):
        self.folds, self.folds_r, self.word_index_to_embeddings_map, self.word_to_indices_map, self.index_to_word_map = \
                    load_train_test_data(dataset)
        self.ling_feat_spmatrix, self.docids = load_ling_features(dataset)
        self.dataset = dataset
            
    def _init_ls(self, feature_type, embeddings_type):
        if self.dataset in self.default_ls_values and feature_type in self.default_ls_values[self.dataset] and \
                                embeddings_type in self.default_ls_values[self.dataset][feature_type]:
            self.default_ls = self.default_ls_values[self.dataset][feature_type][embeddings_type]
        elif 'GP' in self.method:
            self.default_ls = compute_lengthscale_heuristic(feature_type, embeddings_type, self.embeddings,
                                 self.ling_feat_spmatrix, self.docids, self.folds, self.index_to_word_map)
            if self.dataset not in self.default_ls_values:
                self.default_ls_values[self.dataset] = {}
            if feature_type not in self.default_ls_values[self.dataset]:
                self.default_ls_values[self.dataset][feature_type] = {}
            self.default_ls_values[self.dataset][feature_type][embeddings_type] = self.default_ls
        else:
            self.default_ls = []        

    def _set_embeddings(self, embeddings_type):
        if 'word_mean' == embeddings_type and not hasattr(self, 'word_embeddings'):
            self.word_embeddings = load_embeddings(self.word_index_to_embeddings_map)
        elif 'word_mean' != embeddings_type:
            self.word_embeddings = None
            
        if 'skipthoughts' == embeddings_type and not hasattr(self, 'skipthoughts_model'):
            self.skipthoughts_model = load_skipthoughts_embeddings(self.word_to_indices_map)
        elif 'skipthoughts' != embeddings_type:
            self.skipthoughts_model = None
            
        if 'siamese-cbow' == embeddings_type and not hasattr(self, 'siamese_cbow_embeddings'):
            self.siamese_cbow_embeddings = load_siamese_cbow_embeddings(self.word_to_indices_map)
        elif 'siamese-cbow' != embeddings_type:
            self.siamese_cbow_embeddings = None        
        
        if embeddings_type == 'word_mean':
            self.embeddings = self.word_embeddings
        elif embeddings_type == 'skipthoughts':
            self.embeddings = self.skipthoughts_model
        elif embeddings_type == 'siamese-cbow':
            self.embeddings = self.siamese_cbow_embeddings
        else:
            self.embeddings = None    

    def _reload_partial_result(self, resultsfile):
        if not os.path.isfile(resultsfile):
            all_proba = {}
            all_predictions = {}
            all_f = {}
            all_tr_proba = {}
            
            all_target_prefs = {}
            all_target_rankscores = {}
            final_ls = {}
            times = {}
        else:
            with open(resultsfile, 'rb') as fh:
                all_proba, all_predictions, all_f, all_target_prefs, all_target_rankscores, _, times, final_ls, \
                                                                    all_tr_proba = pickle.load(fh, encoding='latin1')
            if all_tr_proba is None:
                all_tr_proba = {}
                
        return all_proba, all_predictions, all_f, all_target_prefs, all_target_rankscores, times, final_ls, all_tr_proba
           
    def run_test(self, feature_type, embeddings_type=None, dataset_increment=0, acc=1.0, subsample_amount=0, 
                 min_no_folds=0, max_no_folds=32, npairs=0, test_on_all_training_pairs=False):

        logging.info("**** Running method %s with features %s, embeddings %s, on dataset %s ****" % (self.method, 
                                                        feature_type, embeddings_type, self.dataset) )
    
        self._set_embeddings(embeddings_type) 
                                                
        self._init_ls(feature_type, embeddings_type)

        resultsfile, results_stem = self._set_resultsfile(feature_type, embeddings_type, acc, dataset_increment)
                       
        all_proba, all_predictions, all_f, all_target_prefs, all_target_rankscores, times, final_ls, all_tr_proba = \
                                                                                self._reload_partial_result(resultsfile)
                
        np.random.seed(121) # allows us to get the same initialisation for all methods/feature types/embeddings

        if os.path.isfile(results_stem + '/foldorder.txt'):
            fold_keys = np.genfromtxt(os.path.expanduser(results_stem + '/foldorder.txt'), dtype=str)
        else:
            fold_keys = list(self.folds.keys())

        for foldidx, self.fold in enumerate(fold_keys):
            if foldidx in all_proba and dataset_increment==0:
                print(("Skipping fold %i, %s" % (foldidx, self.fold)))
                continue
            if foldidx >= max_no_folds or foldidx < min_no_folds:
                print(("Already completed maximum no. folds. Skipping fold %i, %s" % (foldidx, self.fold)))
                continue
            foldresultsfile = results_stem + '/fold%i.pkl' % foldidx
            if foldidx not in all_proba and os.path.isfile(foldresultsfile): 
                if dataset_increment == 0:
                    print(("Skipping fold %i, %s" % (foldidx, self.fold)))
                    continue
                
                with open(foldresultsfile, 'rb') as fh:
                    all_proba[foldidx], all_predictions[foldidx], all_f[foldidx], all_target_prefs[foldidx],\
                    all_target_rankscores[foldidx], _, times[foldidx], final_ls[foldidx], all_tr_proba[foldidx] = \
                                pickle.load(fh, encoding='latin1')
    
            # Get data for this fold --------------------------------------------------------------------------------------
            print(("Fold name ", self.fold))
            a1_train, a2_train, prefs_train, person_train, a1_test, a2_test, prefs_test, person_test,\
                                self.X, uids, utexts = get_noisy_fold_data(self.folds, self.fold, self.docids, acc)                            
            
            # ranking folds
            a_rank_train, scores_rank_train, _, person_rank_train, a_rank_test, scores_rank_test, _, \
                                person_idx_ranktest = get_fold_regression_data(self.folds_r, self.fold, self.docids)
            
            self.load_features(feature_type, embeddings_type, a1_train, a2_train, uids, utexts)
            #items_feat = items_feat[:, :ndebug_features]     
    
            # Subsample training data --------------------------------------------------------------------------------------    
            if npairs == 0:
                npairs_f = len(a1_train)
            else:
                npairs_f = npairs
            nseen_so_far = 0     
                              
            if dataset_increment != 0:
                if foldidx in all_proba and all_proba[foldidx].shape[1] >= float(npairs_f) / dataset_increment:
                    print(("Skipping fold %i, %s" % (foldidx, self.fold)))
                    continue
                nnew_pairs = dataset_increment
            else:
                nnew_pairs = npairs_f
                
            # choose the initial dataset 
            if self.fold in self.initial_pair_subset:    
                pair_subset = self.initial_pair_subset[self.fold]
            elif  dataset_increment != 0:
                pair_subset = np.random.choice(len(a1_train), nnew_pairs, replace=False)
            elif subsample_amount > 0:
                pair_subset = subsample_tr_data(subsample_amount, a1_train, a2_train)                     
            else:
                pair_subset = np.arange(npairs_f)
            # save so we can reuse for another method
            self.initial_pair_subset[self.fold] = pair_subset

            self.verbose = verbose
            self.optimize_hyper = ('noOpt' not in self.method)
                        
    #         with open(modelfile % foldidx, 'r') as fh:
    #             model = pickle.load(fh)
    #             items_feat_test = None
            self.model = None # initial value

            if len(self.default_ls) > 1:
                self.ls_initial = self.default_ls[self.valid_feats]
            else:
                self.ls_initial = self.default_ls

            if '_oneLS' in self.method:
                self.ls_initial = np.median(self.ls_initial)
                logging.info("Selecting a single LS for all features: %f" % self.ls_initial)
            
            logging.info("Starting test with method %s..." % (self.method))
            starttime = time.time()        
            
            unseen_subset = np.ones(len(a1_train), dtype=bool)
    
            # Run the chosen method with active learning simulation if required---------------------------------------------
            while nseen_so_far < npairs_f:
                logging.info('****** Fitting model with %i pairs in fold %i ******' % (len(pair_subset), foldidx))
                
                # get the indexes of data points that are not yet seen        
                if not test_on_all_training_pairs:
                    unseen_subset[pair_subset] = False
                    if dataset_increment == 0: # no active learning, don't need to evaluate the unseen data points
                        unseen_subset[:] = False
    
                # set the current dataset    
                self.a1_train = a1_train[pair_subset]
                self.a2_train = a2_train[pair_subset]
                self.prefs_train = prefs_train[pair_subset]
                self.person_train = person_train
                
                self.a1_test = a1_test
                self.a2_test = a2_test
                self.person_test = person_test
                
                self.a1_unseen = a1_train[unseen_subset]
                self.a2_unseen = a2_train[unseen_subset]
                
                self.a_rank_train = a_rank_train
                self.scores_rank_train = scores_rank_train
                self.person_rank_train = person_rank_train
                                
                self.a_rank_test = a_rank_test
                self.person_rank_test = person_idx_ranktest
                if self.a_rank_test is not None and len(self.person_rank_test) == 0:
                    self.person_rank_test = np.zeros(len(self.a_rank_test)) # if no person IDs, make sure we default to 0
                
                # run the method with the current data subset
                method_runner_fun = self._choose_method_fun(feature_type)
                    
                proba, predicted_f, tr_proba = method_runner_fun()
            
                endtime = time.time() 
                
                # make it the right shape
                proba = np.array(proba)
                if proba.ndim == 2 and proba.shape[1] > 1:
                    proba = proba[:, 1:2]
                elif proba.ndim == 1:
                    proba = proba[:, None]
                predictions = np.round(proba)
                
                if predicted_f is not None:
                    predicted_f = np.array(predicted_f)
                    if predicted_f.ndim == 3:
                        predicted_f = predicted_f[0]
                    if predicted_f.ndim == 1:
                        predicted_f = predicted_f[:, None]
                    
                if tr_proba is not None:
                    tr_proba = np.array(tr_proba)
                    if tr_proba.ndim == 2 and tr_proba.shape[1] > 1:
                        tr_proba = tr_proba[:, 1:2]
                    elif tr_proba.ndim == 1:
                        tr_proba = tr_proba[:, None]
                        
                # get more data
                nseen_so_far += nnew_pairs
                nnew_pairs = dataset_increment#int(np.floor(dataset_increment * npairs_f))
                if nseen_so_far >= npairs_f:
                    # the last iteration possible
                    nnew_pairs = npairs_f - nseen_so_far
                    nseen_so_far = npairs_f    
                else:
                    # don't do this if we have already seen all the data
                    # use predictions at available training points
                    tr_proba = np.array(tr_proba)
                    uncertainty = tr_proba * np.log(tr_proba) + (1-tr_proba) * np.log(1-tr_proba) # negative shannon entropy
                    ranked_pair_idxs = np.argsort(uncertainty.flatten())
                    new_pair_subset = ranked_pair_idxs[:nnew_pairs] # active learning (uncertainty sampling) step
                    new_pair_subset = np.argwhere(unseen_subset)[new_pair_subset].flatten()
                    pair_subset = np.concatenate((pair_subset, new_pair_subset))
                    
                if tr_proba is not None:
                    tr_proba_complete = prefs_train.flatten()[:, np.newaxis] / 2.0
                    tr_proba_complete[unseen_subset] = tr_proba
                    tr_proba = tr_proba_complete
                    
                logging.info("@@@ Completed running fold %i with method %s, features %s, %i data so far, in %f seconds." % (
                    foldidx, self.method, feature_type, nseen_so_far, endtime-starttime) )
                logging.info("Accuracy for fold = %f" % (
                        np.sum(prefs_test[prefs_test != 1] == 2 * predictions.flatten()[prefs_test != 1]
                            ) / float(np.sum(prefs_test != 1))) )
                
                if predicted_f is not None:
                    # print out the pearson correlation
                    logging.info("Pearson correlation for fold = %f" % pearsonr(scores_rank_test, predicted_f.flatten())[0])
                  
                if tr_proba is not None:
                    prefs_unseen = prefs_train[unseen_subset]
                    tr_proba_unseen = tr_proba[unseen_subset]
                    logging.info("Unseen data in the training fold, accuracy for fold = %f" % (
                        np.sum(prefs_unseen[prefs_unseen != 1] == 2 * np.round(tr_proba_unseen).flatten()[prefs_unseen != 1]
                            ) / float(np.sum(prefs_unseen != 1))) )   
                                       
                logging.info("AUC = %f" % roc_auc_score(prefs_test[prefs_test!=1] / 2.0, proba[prefs_test!=1]) )
                # Save the data for later analysis ----------------------------------------------------------------------------
                if hasattr(self.model, 'ls'):
                    final_ls[foldidx] = self.model.ls
                else:
                    final_ls[foldidx] = [0]
                
                # Outputs from the tested method
                if foldidx not in all_proba:
                    all_proba[foldidx] = proba
                    all_predictions[foldidx] = predictions
                    all_f[foldidx] = predicted_f
                    all_tr_proba[foldidx] = tr_proba
                else:
                    all_proba[foldidx] = np.concatenate((all_proba[foldidx], proba), axis=1)
                    all_predictions[foldidx] = np.concatenate((all_predictions[foldidx], predictions), axis=1)
                    if predicted_f is not None:                
                        all_f[foldidx] = np.concatenate((all_f[foldidx], predicted_f), axis=1)
                    if tr_proba is not None:
                        all_tr_proba[foldidx] = np.concatenate((all_tr_proba[foldidx], tr_proba), axis=1)
                
                # Save the ground truth
                all_target_prefs[foldidx] = prefs_test
                if self.folds_r is not None:
                    all_target_rankscores[foldidx] = scores_rank_test
                else:
                    all_target_rankscores[foldidx] = None
                
                # Save the time taken
                times[foldidx] = endtime-starttime                
    
                results = (all_proba[foldidx], all_predictions[foldidx], all_f[foldidx], all_target_prefs[foldidx],
                   all_target_rankscores[foldidx], self.ls_initial, times[foldidx], final_ls[foldidx], 
                   all_tr_proba[foldidx], len(self.a1_train))
                with open(foldresultsfile, 'wb') as fh:
                    pickle.dump(results, fh)
    
                if not os.path.isfile(results_stem + "/foldorder.txt"):
                    save_fold_order(results_stem, self.folds)
                    
                #with open(modelfile % foldidx, 'w') as fh:
                #        pickle.dump(model, fh)
    
            del self.model # release the memory before we try to do another iteration         

    def run_test_set(self, subsample_tr=0, min_no_folds=0, max_no_folds=32, 
                     npairs=0, test_on_train=False):
        # keep these variables around in case we are restarting the script with different method settings and same data.
        for dataset in self.datasets:
            
            self.initial_pair_subset = {} # reset this when we use a different dataset
            
            for self.method in self.methods:
                if self.folds is None or self.dataset != dataset:
                    self._load_dataset(dataset) # reload only if we use a new dataset
               
                if (self.dataset == 'UKPConvArgAll' or self.dataset == 'UKPConvArgStrict' or self.dataset == 'UKPConvArgCrowd_evalAll') \
                    and ('IndPref' in self.method or 'Personalised' in self.method):
                    logging.warning('Skipping method %s on dataset %s because there are no separate worker IDs.' 
                                    % (self.method, self.dataset))
                    continue
                
                for feature_type in self.feature_types:
                    if feature_type == 'embeddings' or feature_type == 'both' or feature_type=='debug':
                        embeddings_to_use = self.embeddings_types
                    else:
                        embeddings_to_use = ['']
                        
                    for embeddings_type in embeddings_to_use:                         
                        self.run_test(feature_type, embeddings_type, dataset_increment=self.dataset_increment, acc=1.0, 
                                subsample_amount=subsample_tr, min_no_folds=min_no_folds, max_no_folds=max_no_folds, 
                                npairs=npairs, test_on_all_training_pairs=test_on_train)
                        
                        logging.info("**** Completed: method %s with features %s, embeddings %s ****" % (self.method, feature_type, 
                                                                               embeddings_type) )
if __name__ == '__main__':

    acc = 1.0
    dataset_increment = 0
         
    datasets = ['UKPConvArgStrict']
    methods = ['SinglePrefGP_weaksprior']
    feature_types = ['both']
    embeddings_types = ['word_mean']

    runner = TestRunner('crowdsourcing_argumentation_expts', datasets, feature_types, embeddings_types, methods,
                            dataset_increment)
    runner.run_test_set(min_no_folds=0, max_no_folds=32)
