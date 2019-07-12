'''
Show the effect of cycles and undecided labels in preference pairs in simple training datasets on:
- GPPL
- SVC
- Ranking using PageRank

We use these three because they have different ways of modelling preference pairs: as noisy observations at both points;
as classifications; as graphs.  

Created on 20 Jul 2017

@author: simpson
'''
import os, sys

from gp_pref_learning import GPPrefLearning

sys.path.append("./python")
sys.path.append("./python/analysis")
sys.path.append("./python/models")
sys.path.append("./python/analysis/habernal_comparison")

sys.path.append(os.path.expanduser("~/git/HeatMapBCC/python"))
sys.path.append(os.path.expanduser("~/git/pyIBCC/python"))

import numpy as np
import matplotlib.pyplot as plt
from tests import load_ling_features, get_noisy_fold_data, load_embeddings, \
                        compute_lengthscale_heuristic, get_mean_embeddings
from data_loading import load_train_test_data
import networkx as nx
from sklearn.svm import SVC

import matplotlib
matplotlib.rcParams.update({'font.size': 16})

def run_pagerank(trainids_a1, trainids_a2, prefs_train):
    G = nx.DiGraph()
    for i in range(len(trainids_a1)):
        if prefs_train[i] == 2:
            G.add_edge(trainids_a1[i], trainids_a2[i])
        elif prefs_train[i] == 0:
            G.add_edge(trainids_a2[i], trainids_a1[i])
            
    rank_dict = nx.pagerank_numpy(G)
    rankscores = np.zeros(len(rank_dict))
    rankscores[list(rank_dict.keys())] = list(rank_dict.values())
    return rankscores
    
def run_svm(trainids_a1, trainids_a2, prefs_train, items_feat, testids_a1, testids_a2):    
    svc = SVC(probability=True)
    
    prefs_train = np.copy(prefs_train)
    #ignore the undecided labels
    trainids_a1 = trainids_a1[prefs_train!=1]
    trainids_a2 = trainids_a2[prefs_train!=1]
    prefs_train = prefs_train[prefs_train!=1]
    
    svc.fit(
        #np.concatenate((items_feat[trainids_a1, :], items_feat[trainids_a2, :]), axis=1),
        #np.array(prefs_train) / 2.0)
        np.concatenate((np.concatenate((items_feat[trainids_a1, :], items_feat[trainids_a2, :]), axis=1), 
            np.concatenate((items_feat[trainids_a2, :], items_feat[trainids_a1, :]), axis=1)), axis=0), 
            np.concatenate((np.array(prefs_train) / 2.0, 1 - np.array(prefs_train) / 2.0)) )
    #results['SVM'] = svc.decision_function(targets_single_arr)
    #proba = svc.predict_proba(np.concatenate((items_feat[testids_a1, :], items_feat[testids_a2, :]), axis=1))
    pred_svm = svc.predict(np.concatenate((items_feat[testids_a1, :], items_feat[testids_a2, :]), axis=1))
    #return proba[:, np.argwhere(np.array(svc.classes_)==1)[0][0]]
    return pred_svm

def plot_probas(total_p, label, outputdir, N, vmin=0, vmax=1):
    mean_p = total_p / float(nrepeats)

    # Plot classifications of all pairs as a coloured 3x3 table
    plt.figure(figsize=(4,3))
    data = mean_p.reshape(N, N) # do 1 - to get the preference for the argument along x axis over arg along y axis 
    im = plt.imshow(data, interpolation='nearest', vmin=vmin, vmax=vmax, cmap=plt.cm.get_cmap('hot'))
    plt.grid('on')
    plt.title('%s -- Predicted Preferences: p(arg_x > arg_y)' % label)
    plt.xlabel('ID of arg_x')
    plt.ylabel('ID of arg_y')
    plt.xticks(list(range(N)))
    plt.yticks(list(range(N)))
    plt.colorbar(im, fraction=0.046, pad=0.04, shrink=0.9)  
    
    plt.savefig(outputdir + '/' + label + '_probas.pdf')  
    
def plot_scores(total_f, var_f, label, outputdir, sample_objs, obj_labels, methodnum, max_normalize_val, fig=None):
    # normalise it
    total_f /= max_normalize_val
    mean_f = total_f / float(nrepeats)
    
    # Plot the latent function values for a, b, and c as a bar chart
    if fig is None:
        fig = plt.figure(figsize=(3,3))
    else:
        plt.figure(fig.number)
        
    cols = ['steelblue', 'maroon', 'lightblue']
    
    if methodnum == 0:
        plt.plot([-0.5, len(sample_objs)-0.5], [0, 0], color='black')
    
    if var_f is not None:
        var_f /= max_normalize_val**2
        var_f /= float(nrepeats)**2
        plt.bar(sample_objs - (0.47*methodnum) + 0.235, mean_f.flatten(), 0.45, color=cols[methodnum], 
                yerr=np.sqrt(var_f).flatten(), label=label)
    else:
        plt.bar(sample_objs - (0.47*methodnum) + 0.235, mean_f.flatten(), 0.45, color=cols[methodnum], label=label)
        
    plt.gca().set_xticks(sample_objs)
    plt.gca().set_xticklabels(obj_labels)
    #plt.gca().spines['bottom'].set_position('zero')
    plt.legend(loc='best')
    
    if var_f is None:
        plt.title('%s: Estimated latent function values' % label)
    else:
        plt.title('%s: Mean latent function values with STD error bars' % label)
    plt.grid('on')
    plt.xlim([-0.5, len(sample_objs)-0.5])
    
    
    plt.savefig(outputdir + '/' + label + '_scores.pdf')
    
    return fig
    
def plot_arg_graph(prefs_train, sample_objs, obj_labels, outputdir, label):
    # Plot the training data graph.                                        
    
    sample_objs_ycoords = np.mod(sample_objs, 2)
                                
    plt.figure(figsize=(4,3))  
        
    for p in range(len(prefs_train)):  
        if prefs_train[p] == 0:
            a1 = trainids_a2[p]
            a2 = trainids_a1[p]
            headwidth = 0.1
            shift = 0
        elif prefs_train[p] == 2:
            a1 = trainids_a1[p]
            a2 = trainids_a2[p]
            headwidth = 0.1
            shift = 0
        else: # shift the line slightly
            a1 = trainids_a1[p]
            a2 = trainids_a2[p]
            headwidth = 0
            shift = 0.25 * (p / float(len(prefs_train) + 1) - 0.5)
                                                              
        plt.arrow(sample_objs[a1], sample_objs_ycoords[a1] + shift, 
                  (sample_objs[a2] - sample_objs[a1]) / 2.0, 
                  (sample_objs_ycoords[a2] - sample_objs_ycoords[a1]) / 2.0,
                  color='black', head_width=headwidth)            
        plt.arrow(sample_objs[a1] + (sample_objs[a2]-sample_objs[a1]) / 2.0, 
                  shift + sample_objs_ycoords[a1] + (sample_objs_ycoords[a2] -
                                                         sample_objs_ycoords[a1]) / 2.0, 
                  (sample_objs[a2]-sample_objs[a1]) / 2.0, 
                  (sample_objs_ycoords[a2] - sample_objs_ycoords[a1]) / 2.0,
                  color='black')

    
    plt.scatter(sample_objs, sample_objs_ycoords, marker='o', s=400, color='black')
    
    for obj in range(len(sample_objs)):
        plt.text(sample_objs[obj]+0.18, sample_objs_ycoords[obj]+0.08, obj_labels[obj])
                
    plt.xlim([-1.0, len(sample_objs)])
    plt.ylim([-0.5, np.max(sample_objs_ycoords) + 0.5])        
    plt.axis('off')
    plt.title('Argument Preference Graph')    
        
    plt.savefig(outputdir + '/' + label + '_arg_graph.pdf')  

def run_test(label, trainids_a1, trainids_a2, prefs_train, nrepeats):

    trainids_a1 = np.array(trainids_a1)
    trainids_a2 = np.array(trainids_a2)
    
    total_f_gppl = 0
    total_p_gppl = 0
    total_v_gppl = 0
    total_p_svm = 0    
    total_f_pagerank = 0
    
    output_dir = os.path.expanduser(
        './documents/pref_learning_for_convincingness/figures/cycles_demo2/')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    output_dir +=  label + '/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)    
    
    for r in range(nrepeats):
        print(('*** Running repeat %i ***' % r))
        sample_objs = np.sort(np.unique(np.concatenate((trainids_a1, trainids_a2))))
        obj_labels = []
        for arg in sample_objs:
            obj_labels.append('arg%i' % arg)

        # test all possible pairs!
        N = len(sample_objs)
        testids_a1 = np.tile(sample_objs[:, None], (1, N)).flatten()
        testids_a2 = np.tile(sample_objs[None, :], (N, 1)).flatten()
    
        if r == 0:            
            plot_arg_graph(prefs_train, sample_objs, obj_labels, output_dir, 'arggraph')
    
        # Run GPPL.
        model = GPPrefLearning(ninput_features=items_feat.shape[1], ls_initial=default_ls_value, verbose=False,
                    shape_s0=2.0, rate_s0=200.0, rate_ls = 1.0 / np.mean(default_ls_value), use_svi=True,
                    ninducing=500, max_update_size=200, kernel_combination='*', forgetting_rate=0.7,
                    delay=1.0)
        model.fit(trainids_a1, trainids_a2, items_feat, np.array(prefs_train, dtype=float)-1,
                  optimize=False, input_type='zero-centered')
        proba = model.predict(None, testids_a1, testids_a2, reuse_output_kernel=True, return_var=False)
        predicted_f, _ = model.predict_f(None, sample_objs)

        #fold, None, method, , ,


        # Flip the latent function because GPPL treats 2.0 as a preference for the first item in the pair,
        # whereas in our data it is a preference for the second item. If we don't flip, the latent function correlates
        # with a ranking (the lower the better). The predicted class labels don't require flipping. 
        predicted_f = -predicted_f   
        _, f_var = model.predict_f(None, sample_objs)
        
        total_p_gppl += proba
        total_v_gppl += f_var
        total_f_gppl += predicted_f    
        
        # Run SVC
        proba_svm = run_svm(trainids_a1, trainids_a2, prefs_train, items_feat, testids_a1, testids_a2)
        total_p_svm += proba_svm
        
        print('agreement SVM and GPPL: %f' % (1 - np.sum(np.abs(np.round(proba_svm) - np.round(proba.flatten()))) / float(proba.size)))
        
        # Run PageRank        
        pagerank_f = run_pagerank(trainids_a1, trainids_a2, prefs_train)      
        total_f_pagerank += pagerank_f
        
    max_val = np.max(total_f_gppl)
    plot_probas(total_p_gppl, 'GPPL', output_dir, N)
    fig = plot_scores(total_f_gppl, total_v_gppl, 'GPPL', output_dir, sample_objs, obj_labels, 0, max_val)
    plot_probas(total_p_svm, 'SVM', output_dir, N)
    max_val = np.max(total_f_pagerank)
    plot_scores(total_f_pagerank, None, 'PageRank', output_dir, sample_objs, obj_labels, 1, max_val, fig=fig)
    
    #plt.show()
    plt.close('all')
    
if __name__ == '__main__':
    # start by loading some realistic feature data. We don't need the preference pairs -- we'll make them up!    
    feature_type = 'embeddings'
    embeddings_type = 'word_mean'
    dataset = 'UKPConvArgStrict'
    method = 'SinglePrefGP_weaksprior_noOpt'
    
    if 'folds' not in globals():
        # load some example data.
        folds, folds_regression, word_index_to_embeddings_map, word_to_indices_map, index_to_word_map = \
                                                                                    load_train_test_data(dataset)    
    
    ling_feat_spmatrix, docids = load_ling_features(dataset)
    embeddings = load_embeddings(word_index_to_embeddings_map)

    
    if 'default_ls_values' not in globals():
        default_ls_values = {}
        
    if dataset in default_ls_values and feature_type in default_ls_values[dataset] and \
            embeddings_type in default_ls_values[dataset][feature_type]:
        default_ls_value = default_ls_values[dataset][feature_type][embeddings_type]
    elif 'GP' in method:
        default_ls_value = compute_lengthscale_heuristic(feature_type, embeddings_type, embeddings,
                             ling_feat_spmatrix, docids, folds, index_to_word_map)
        if dataset not in default_ls_values:
            default_ls_values[dataset] = {}
        if feature_type not in default_ls_values[dataset]:
            default_ls_values[dataset][feature_type] = {}
        default_ls_values[dataset][feature_type][embeddings_type] = default_ls_value
    else:
        default_ls_value = []

    fold = list(folds.keys())[0]
    print(("Fold name ", fold))
    trainids_a1, trainids_a2, prefs_train, personIDs_train, testids_a1, testids_a2, prefs_test, personIDs_test,\
                        X, uids, utexts = get_noisy_fold_data(folds, fold, docids, 1.0)

    items_feat = get_mean_embeddings(embeddings, X)
    ndims = items_feat.shape[1]
    # Generate simple training data containing a->b, b->c, c->a cycle.
    nrepeats = 25

    trainids_a1 = [0, 1, 2, 3, 4]
    trainids_a2 = [2, 2, 0, 2, 2]
    prefs_train = [2, 2, 0, 2, 2]
    run_test('single_hub', trainids_a1, trainids_a2, prefs_train, nrepeats)

    trainids_a1 = [0, 1, 2, 3, 4]
    trainids_a2 = [1, 2, 0, 4, 3]
    prefs_train = [0, 0, 2, 0, 2]
    run_test('no_cycle', trainids_a1, trainids_a2, prefs_train, nrepeats)
 
    trainids_a1 = [0, 1, 2, 3, 4]
    trainids_a2 = [1, 2, 0, 4, 3]
    prefs_train = [0, 0, 0, 0, 2]
    run_test('simple_cycle', trainids_a1, trainids_a2, prefs_train, nrepeats)
      
    trainids_a1 = [0, 1, 2, 0, 3]
    trainids_a2 = [1, 2, 0, 3, 2]
    prefs_train = [0, 0, 0, 0, 0]
    run_test('double_cycle', trainids_a1, trainids_a2, prefs_train, nrepeats)
      
    trainids_a1 = [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4]
    trainids_a2 = [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3]
    prefs_train = [0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2]
    run_test('undecided', trainids_a1, trainids_a2, prefs_train, nrepeats)