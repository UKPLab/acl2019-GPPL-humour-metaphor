'''
Run tests on the Lukin dataset here. To demonstrate how to use the PreferenceComponents method, 
we do the following simple test:
- load the dataset
- train on the whole dataset
- for a set of test users (for demonstration, we just sample randomly from training set), predict latent features
- for a set of test arguments (for demonstration, we sample pairs randomly from training set), predict latent features
- for a set of test pairs (users + arguments chosen at random from training set), predict the preference rating
- for a set of test triples (users + argument pairs chosen at random from the training set), predict the pairwise label    

TODO: move user analysis stuff to a separate script
TODO: switch to 10-fold cross validation instead of training on all
TODO: Add missing prior belief data
TODO: Check whether the lmh labels are correctly converted to preferences -- in some cases they should be flipped

Created on Sep 25, 2017

@author: simpson
'''

import numpy as np, os, sys
from sklearn.metrics.classification import accuracy_score
import logging
import time
logging.basicConfig(level=logging.DEBUG)

sys.path.append("./python")
sys.path.append("./python/analysis")
sys.path.append("./python/models")
sys.path.append("./python/analysis/lukin_comparison")

sys.path.append(os.path.expanduser("~/git/HeatMapBCC/python"))
sys.path.append(os.path.expanduser("~/git/pyIBCC/python"))

from collab_pref_learning_svi import CollabPrefLearningSVI
from collab_pref_learning_vb import CollabPrefLearningVB
from gp_classifier_vb import compute_median_lengthscales

use_entrenched = True

if __name__ == '__main__':
    np.random.seed(1212)
    
    #load the dataset -----------------------------------------------------------------------------------
    arg_data = np.genfromtxt('./data/lukin/arguments2.csv', dtype=float, delimiter=',', skip_header=1)
    arg_ids = arg_data[:, 0].astype(int)
    item_feat = arg_data[:, 1:]
    
    user_data = np.genfromtxt('./data/lukin/users.csv', dtype=float, delimiter=',', skip_header=1)
    user_ids = user_data[:, 0].astype(int)
    person_feat = user_data[:, 1:]
    npersonality_feats = person_feat.shape[1]
    
    pair_data = np.genfromtxt('./data/lukin/ratings2.csv', dtype=float, delimiter=',', skip_header=1)
    #debug_subset = np.arange(200)
    #pair_data = pair_data[debug_subset, :]
    
    # should we use the 'entrenched' column as an additional user feature?
    if use_entrenched:
        # double the person feature vector -- each person appears twice, once for entrenched, once for not entrenched
        Npeople = person_feat.shape[0]
        person_feat = np.concatenate((person_feat, person_feat), axis=0)
        entrenched_feat = np.zeros((Npeople*2, 1))
        entrenched_feat[:Npeople] = 1
        person_feat = np.concatenate((person_feat, entrenched_feat), axis=1)
        
        entrenched_labels = np.genfromtxt('./data/lukin/ratings2.csv', dtype=bool, delimiter=',', skip_header=1)[:, 3]
        #entrenched_labels = entrenched_labels[debug_subset]
        personIDs_train = np.array([np.argwhere(user_ids==uid)[0][0] + (Npeople*entrenched_labels[i]) for i, uid in 
                                    enumerate(pair_data[:, 0].astype(int))])
        Npeople = Npeople * 2
    else:
        personIDs_train = np.array([np.argwhere(user_ids==uid)[0][0] for uid in pair_data[:, 0].astype(int)])
    trainids_a1 = np.array([np.argwhere(arg_ids==aid)[0][0] for aid in pair_data[:, 1].astype(int)])
    trainids_a2 = np.array([np.argwhere(arg_ids==aid)[0][0] for aid in pair_data[:, 2].astype(int)])

    #prefs_train = pair_data[:, 6] # use the norm labels
    #prefs_train = pair_data[:, 4] # use the lmh labels
    prefs_train = pair_data[:, 5] # use the lh labels
    
    # Training ---------------------------------------------------------------------------------------------
    #train on the whole dataset
    ndims = item_feat.shape[1]
    ls_initial = compute_median_lengthscales(item_feat) / 10.0
    person_ls_initial = compute_median_lengthscales(person_feat) / 10.0
    
    #nfactors = person_feat.shape[0]
    #if item_feat.shape[0] < person_feat.shape[0]:
    #    nfactors = item_feat.shape[0] # use smallest out of N and Npeople
    nfactors = 10
    model = CollabPrefLearningSVI(nitem_features=ndims, ls=ls_initial, lsy=person_ls_initial, shape_s0=2,
                                  rate_s0=200/float(nfactors), verbose=True, nfactors=nfactors,
                                  rate_ls = 1.0 / np.mean(ls_initial), use_common_mean_t=False)
    model.max_iter = 1000
    model.fit(personIDs_train, trainids_a1, trainids_a2, item_feat, prefs_train, person_feat, optimize=False, 
              nrestarts=1, input_type='zero-centered', use_lb=True)
        
    # sanity check: test on the training data
    trpred = model.predict(personIDs_train, trainids_a1, trainids_a2, item_feat, person_feat)
    tracc = accuracy_score(np.round((prefs_train + 1) / 2.0), np.round(trpred))
    print("The model was trained. Testing on the training data gives an accuracy of %.4f" % tracc)
        
    # Get some test data ------------------------------------------------------------------------------------
    
    # ***** Section to replace with a matrix of real test data *****
    #test_arg_ids = np.arange(100) # PLACEHOLDER -- replace this with the real document IDs loaded from file
    #test_item_feat = item_feat[test_arg_ids, :] # PLACEHOLDER -- replace this with the real test document IDs loaded from file
    
    # load test items from file
    test_data = np.genfromtxt('./data/lukin/test_input.csv', dtype=float, delimiter=',', skip_header=1)
    test_arg_ids = test_data[:, 0]
    test_item_feat = test_data[:, 1:]
    
    # ***** End *****
    
    Ntest = len(test_arg_ids) # keep this
    testids = np.arange(Ntest) # keep this -- this is the index into the test_item_feat. In this case, we test all the items
    
    #for a set of test arguments (for demonstration, we sample pairs randomly from training set), predict latent features
    #w = model.predict_item_feats(testids, item_feat)
    print(np.max(model.w))
    print(np.min(model.w))
    print(np.max(model.y))
    print(np.min(model.y))
    print(np.max(model.t))
    print(np.min(model.t))

    # Generate test people with all combinations of values at discrete points -----------------------------------------
     
    # Given the argument's latent features, determine the observed features of the most convinced user
    # Chosen method: create a set of simulated users with prototypical personality features
#     feat_min = 1
#     feat_max = 7
#     nfeat_vals = feat_max - feat_min + 1
#     feat_range = nfeat_vals**5 # number of different feature combinations, assuming integer values for personality traits
#     Npeople = feat_range
#  
#     test_people = np.arange(feat_range)
#     test_person_feat = np.zeros((feat_range, npersonality_feats))
#     f_val = np.zeros(test_person_feat.shape[1]) + feat_min
#          
#     for p in range(feat_range):
#         for f in range(npersonality_feats):
#             test_person_feat[p, f] = f_val[f]
#              
#             if np.mod(p+1, (nfeat_vals ** f) ) == 0:
#                 f_val[f] = np.mod(f_val[f], nfeat_vals) + 1
#  
#     testids = np.tile(testids[:, None], (1, feat_range))
#     test_people = np.tile(test_people[None, :], (Ntest, 1))
#  
#     if use_entrenched:
#         test_person_feat = np.concatenate((test_person_feat, test_person_feat), axis=0)
#         entrenched_feat = np.zeros((Npeople*2, 1))
#         entrenched_feat[:Npeople] = 1
#         test_person_feat = np.concatenate((test_person_feat, entrenched_feat), axis=1)
#          
#         test_people = np.concatenate((test_people, test_people + feat_range), axis=1)
#         testids = np.concatenate((testids, testids), axis=1)
#          
#         Npeople = Npeople * 2
    
#     # Reuse the training people (i.e. real profiles) for testing -------------------------------------------------------
    test_person_feat = person_feat    
    Npeople = test_person_feat.shape[0]
    test_people = np.arange(Npeople)
    test_people = np.tile(test_people[None, :], (Ntest, 1)) # test each person against each item
    testids = np.tile(testids[:, None], (1, Npeople))

    # predict the ratings from each of the simulated people
    npairs = Npeople * Ntest
    predicted_f = np.zeros((Ntest, Npeople))
    # do it in batches of 500 because there are too many people
    batchsize = 2000
    nbatches = int(np.ceil(npairs / float(batchsize)))
    
    for b in range(nbatches):
        print("Predicting simulated users in batch %i of %i" % (b, nbatches))
        start = batchsize * b
        fin = batchsize * (b + 1)
        if fin > npairs:
            fin = npairs
            
        rows, cols = np.unravel_index(np.arange(start, fin), dims=(Ntest, Npeople))
        predicted_f[rows, cols] = model.predict_f_item_person(test_people[rows, cols],
                                                 testids[rows, cols], 
                                                 test_item_feat, test_person_feat)
    
    # who had the highest preference?
    max_people_idx = np.argmax(predicted_f, axis=1)
        
    # get the features of the max people
    max_people_feats = test_person_feat[max_people_idx, :]
    
    out_data = np.concatenate((test_arg_ids[:, None], max_people_feats), axis=1)
    
    # save to file along with original argument IDs from the test data file
    fmt = '%i, '
    header = 'argID,openness,conscientiousness,extroversion,agreeableness,neuroticism'
    for f in range(npersonality_feats):
        fmt += '%f, '

    if use_entrenched:
        fmt += '%i'
        header += ',entrenched'
        
    if not os.path.isdir('./results/lukin'):
        os.mkdir('./results/lukin')
    timestamp = time.strftime('%Y_%m_%d_%H_%M_%S')
    np.savetxt('./results/lukin/personalities_for_args_%s.csv' % timestamp, out_data, fmt, ',', header=header)
