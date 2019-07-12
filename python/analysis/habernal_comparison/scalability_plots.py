'''
Created on Jan 8, 2018

@author: simpson
'''

import os
import numpy as np
from compute_metrics import load_results_data, get_fold_data
import pickle
from sklearn.metrics.classification import accuracy_score
import matplotlib.pyplot as plt
import compute_metrics
from matplotlib import gridspec

figure_save_path = './documents/pref_learning_for_convincingness/figures/scalability'
if not os.path.isdir(figure_save_path):
    os.mkdir(figure_save_path)

if __name__ == '__main__':
    
    if 'expt_settings' not in globals():
        expt_settings = {}
        expt_settings['dataset'] = None
        expt_settings['folds'] = None 
    expt_settings['foldorderfile'] = None    
    data_root_dir = os.path.expanduser("~/data/personalised_argumentation/")
    resultsfile_template = 'habernal_%s_%s_%s_%s_acc%.2f_di%.2f'
        
    expt_settings['dataset'] = 'UKPConvArgStrict'
    expt_settings['acc'] = 1.0
    expt_settings['di'] = 0.0
    compute_metrics.max_no_folds = 32
        
    # Create a plot for the runtime/accuracy against M + include other methods with ling + Glove features
    methods =  ['SinglePrefGP_noOpt_weaksprior_M2', 'SinglePrefGP_noOpt_weaksprior_M10', 
                'SinglePrefGP_noOpt_weaksprior_M100', 'SinglePrefGP_noOpt_weaksprior_M200', 'SinglePrefGP_noOpt_weaksprior_M300',
                'SinglePrefGP_noOpt_weaksprior_M400', 'SinglePrefGP_noOpt_weaksprior_M500',  
                'SinglePrefGP_noOpt_weaksprior_M600', 'SinglePrefGP_noOpt_weaksprior_M700', 
                'SVM', 'BI-LSTM', 'SinglePrefGP_weaksprior', ]
    expt_settings['feature_type'] = 'both'
    expt_settings['embeddings_type'] = 'word_mean'
    
    docids = None
    
    dims_methods = np.array(['SinglePrefGP_noOpt_weaksprior_M500', 'SVM', 'BI-LSTM', 'SinglePrefGP_weaksprior'])
    runtimes_dims = np.zeros((len(dims_methods), 4))
    
    runtimes_both = np.zeros(len(methods))
    acc_both = np.zeros(len(methods))
    
    for m, expt_settings['method'] in enumerate(methods): 
        print("Processing method %s" % expt_settings['method'])

        data, nFolds, resultsdir, resultsfile = load_results_data(data_root_dir, resultsfile_template, 
                                                                          expt_settings)
        
        acc_m = np.zeros(nFolds)
        runtimes_m = np.zeros(nFolds)
        
        for f in range(nFolds):
            print("Processing fold %i" % f)
            if expt_settings['fold_order'] is None: # fall back to the order on the current machine
                if expt_settings['folds'] is None:
                    continue
                fold = list(expt_settings['folds'].keys())[f]
            else:
                fold = expt_settings['fold_order'][f] 
                if fold[-2] == "'" and fold[0] == "'":
                    fold = fold[1:-2]
                elif fold[-1] == "'" and fold[0] == "'":
                    fold = fold[1:-1]  
                expt_settings['fold_order'][f] = fold
                                          
            # look for new-style data in separate files for each fold. Prefer new-style if both are found.
            foldfile = resultsdir + '/fold%i.pkl' % f
            if os.path.isfile(foldfile):
                with open(foldfile, 'rb') as fh:
                    data_f = pickle.load(fh, encoding='latin1')
            else: # convert the old stuff to new stuff
                if data is None:
                    min_folds = f+1
                    print('Skipping fold with no data %i' % f)
                    print("Skipping results for %s, %s, %s, %s" % (expt_settings['method'], 
                                                                   expt_settings['dataset'], 
                                                                   expt_settings['feature_type'], 
                                                                   expt_settings['embeddings_type']))
                    print("Skipped filename was: %s, old-style results file would be %s" % (foldfile, 
                                                                                            resultsfile))
                    continue
                
                if not os.path.isdir(resultsdir):
                    os.mkdir(resultsdir)
                data_f = []
                for thing in data:
                    if f in thing:
                        data_f.append(thing[f])
                    else:
                        data_f.append(thing)
                with open(foldfile, 'wb') as fh:
                    pickle.dump(data_f, fh)  
                              
            gold_disc, pred_disc, gold_prob, pred_prob, gold_rank, pred_rank, pred_tr_disc, \
                                        pred_tr_prob, postprocced = get_fold_data(data_f, f, expt_settings)
                                        
            acc_m[f] = accuracy_score(gold_disc[gold_disc!=1], pred_disc[gold_disc!=1])
            runtimes_m[f] = data_f[6]
            
        acc_m = acc_m[acc_m>0]
        runtimes_m = runtimes_m[runtimes_m>0]            
            
        if len(acc_m):
            acc_both[m] = np.mean(acc_m)
            runtimes_both[m] = np.mean(runtimes_m)
            if expt_settings['method'] in dims_methods:
                m_dims = dims_methods == expt_settings['method']
                runtimes_dims[m_dims, 3] = runtimes_both[m]
      
    expt_settings['feature_type'] = 'embeddings'
        
    runtimes_emb = np.zeros(len(methods))
    acc_emb = np.zeros(len(methods))
    
    for m, expt_settings['method'] in enumerate(methods): 
        print("Processing method %s" % expt_settings['method'])

        data, nFolds, resultsdir, resultsfile = load_results_data(data_root_dir, resultsfile_template, 
                                                                          expt_settings)
        
        acc_m = np.zeros(nFolds)
        runtimes_m = np.zeros(nFolds)
        
        for f in range(nFolds):
            print("Processing fold %i" % f)
            if expt_settings['fold_order'] is None: # fall back to the order on the current machine
                if expt_settings['folds'] is None:
                    continue
                fold = list(expt_settings['folds'].keys())[f]
            else:
                fold = expt_settings['fold_order'][f] 
                if fold[-2] == "'" and fold[0] == "'":
                    fold = fold[1:-2]
                elif fold[-1] == "'" and fold[0] == "'":
                    fold = fold[1:-1]  
                expt_settings['fold_order'][f] = fold
                                                          
            # look for new-style data in separate files for each fold. Prefer new-style if both are found.
            foldfile = resultsdir + '/fold%i.pkl' % f
            if os.path.isfile(foldfile):
                with open(foldfile, 'rb') as fh:
                    data_f = pickle.load(fh, encoding='latin1')
            else: # convert the old stuff to new stuff
                if data is None:
                    min_folds = f+1
                    print('Skipping fold with no data %i' % f)
                    print("Skipping results for %s, %s, %s, %s" % (expt_settings['method'], 
                                                                   expt_settings['dataset'], 
                                                                   expt_settings['feature_type'], 
                                                                   expt_settings['embeddings_type']))
                    print("Skipped filename was: %s, old-style results file would be %s" % (foldfile, 
                                                                                            resultsfile))
                    continue
                
                if not os.path.isdir(resultsdir):
                    os.mkdir(resultsdir)
                data_f = []
                for thing in data:
                    if f in thing:
                        data_f.append(thing[f])
                    else:
                        data_f.append(thing)
                with open(foldfile, 'wb') as fh:
                    pickle.dump(data_f, fh)  
                 
            fold = expt_settings['fold_order'][f]
            if fold[-2] == "'" and fold[0] == "'":
                fold = fold[1:-2]
            elif fold[-1] == "'" and fold[0] == "'":
                fold = fold[1:-1]  
            expt_settings['fold_order'][f] = fold
                         
            gold_disc, pred_disc, gold_prob, pred_prob, gold_rank, pred_rank, pred_tr_disc, \
                                        pred_tr_prob, postprocced = get_fold_data(data_f, f, expt_settings)
                                        
            acc_m[f] = accuracy_score(gold_disc[gold_disc!=1], pred_disc[gold_disc!=1])
            runtimes_m[f] = data_f[6]
            
        acc_m = acc_m[acc_m>0]
        runtimes_m = runtimes_m[runtimes_m>0]
                
        if len(acc_m):
            acc_emb[m] = np.mean(acc_m)
            runtimes_emb[m] = np.mean(runtimes_m)
            if expt_settings['method'] in dims_methods:
                m_dims = dims_methods == expt_settings['method']
                runtimes_dims[m_dims, 1] = runtimes_emb[m]          
        
    # First plot: M versus Runtime and Accuracy for 32310 features -----------------------------------------------------
    
    fig1, ax1 = plt.subplots(figsize=(5,4))
    x_gppl = np.array([2, 10, 100, 200, 300, 400, 500, 600, 700])
    h1, = ax1.plot(x_gppl, runtimes_both[:-3], color='blue', marker='o', label='runtime', 
                   linewidth=2, markersize=8)  
    ax1.set_ylabel('Runtime (s)')
    plt.xlabel('No. Inducing Points, M')
    ax1.grid('on', axis='y')
    ax1.spines['left'].set_color('blue')
    ax1.tick_params('y', colors='blue')
    ax1.yaxis.label.set_color('blue')
    
    ax1_2 = ax1.twinx()
    h2, = ax1_2.plot(x_gppl, acc_both[:-3], color='black', marker='x', label='accuracy', 
                     linewidth=2, markersize=8)
    #ax1_2.set_ylabel('Accuracy')
    leg = plt.legend(handles=[h1, h2], loc='lower right')
    leg.get_texts()[0].set_color('blue')
    
    plt.tight_layout()    
    plt.savefig(figure_save_path + '/num_inducing_32310_features.pdf')        
    
    # Second plot: M versus runtime and accuracy for 300 features ------------------------------------------------------

    fig1, ax2 = plt.subplots(figsize=(5,4))
    h1, = ax2.plot(x_gppl, runtimes_emb[:-3], color='blue', marker='o', label='runtime', 
                   linewidth=2, markersize=8)
    plt.xlabel('No. Inducing Points, M')
    ax2.grid('on', axis='y')   
    #ax2.set_ylabel('Runtime (s)')
    ax2.spines['left'].set_color('blue')
    ax2.tick_params('y', colors='blue')
    ax2.yaxis.label.set_color('blue')    
            
    ax2_2 = ax2.twinx()
    h2, = ax2_2.plot(x_gppl, acc_emb[:-3], color='black', marker='x', label='accuracy', 
                     linewidth=2, markersize=8)
    ax2_2.set_ylabel('Accuracy')
    leg = plt.legend(handles=[h1, h2], loc='lower right')
    leg.get_texts()[0].set_color('blue')
    
    plt.tight_layout()    
    plt.savefig(figure_save_path + '/num_inducing_300_features.pdf')    
    
    # Third plot: training set size N versus runtime (with Glove features) ---------------------------------------------
    
    expt_settings['feature_type'] = 'embeddings'
    methods = ['SinglePrefGP_noOpt_weaksprior_M100', 'SinglePrefGP_noOpt_weaksprior_M0', 
                'SVM_small', 'BI-LSTM'] 
        
    Nvals = [50, 100, 200, 300, 400, 500]
    runtimes_N = np.zeros((len(methods), len(Nvals)))
    
    for n, N in enumerate(Nvals):
        foldername = 'crowdsourcing_argumentation_expts_%i/' % N
        
        for m, expt_settings['method'] in enumerate(methods): 
            print("Processing method %s" % expt_settings['method'])
    
            data, nFolds, resultsdir, resultsfile = load_results_data(data_root_dir, 
                                          resultsfile_template, expt_settings, foldername)

            runtimes_m = np.zeros(nFolds)
            
            for f in range(nFolds):
                print("Processing fold %i" % f)
                fold = expt_settings['fold_order'][f] 
                if fold[-2] == "'" and fold[0] == "'":
                    fold = fold[1:-2]
                elif fold[-1] == "'" and fold[0] == "'":
                    fold = fold[1:-1]  
                expt_settings['fold_order'][f] = fold
                                              
                # look for new-style data in separate files for each fold. Prefer new-style if both are found.
                foldfile = resultsdir + '/fold%i.pkl' % f
                if os.path.isfile(foldfile):
                    with open(foldfile, 'rb') as fh:
                        data_f = pickle.load(fh, encoding='latin1')
                else: # convert the old stuff to new stuff
                    if data is None:
                        min_folds = f+1
                        print('Skipping fold with no data %i' % f)
                        print("Skipping results for %s, %s, %s, %s" % (expt_settings['method'], 
                                                                       expt_settings['dataset'], 
                                                                       expt_settings['feature_type'], 
                                                                       expt_settings['embeddings_type']))
                        print("Skipped filename was: %s, old-style results file would be %s" % (foldfile, 
                                                                                                resultsfile))
                        continue
                    
                    if not os.path.isdir(resultsdir):
                        os.mkdir(resultsdir)
                    data_f = []
                    for thing in data:
                        if f in thing:
                            data_f.append(thing[f])
                        else:
                            data_f.append(thing)
                    with open(foldfile, 'wb') as fh:
                        pickle.dump(data_f, fh)  
                                  
                runtimes_m[f] = data_f[6]
                
            runtimes_N[m, n] = np.mean(runtimes_m)
    
    
    fig3, ax3 = plt.subplots(figsize=(5,4))
    
    ax3.plot(Nvals, runtimes_N[0], label='GPPL M=100', marker='o', linewidth=2, markersize=8)
    ax3.plot(Nvals, runtimes_N[1], label='GPPL no SVI', marker='x', linewidth=2, markersize=8)
    ax3.plot(Nvals, runtimes_N[2], label='SVM', marker='>', color='black', linewidth=2, markersize=8)
    ax3.plot(Nvals, runtimes_N[3], label='BiLSTM', marker='^', color='red', linewidth=2, markersize=8)
     
    ax3.set_xlabel('N_tr (no. arguments in training set)')
    ax3.set_ylabel('Runtime (s)')
    ax3.yaxis.grid('on')
    ax3.set_ylim(-5, 205)
    plt.legend(loc='center')
    
    plt.tight_layout()    
    plt.savefig(figure_save_path + '/num_arguments.pdf')    
    
    # Fourth plot: no. features versus runtime -------------------------------------------------------------------------
    expt_settings['feature_type'] = 'debug'
    for n, dim in enumerate(['30feats', '', '3000feats']):
        foldername = 'crowdsourcing_argumentation_expts_%s/' % dim
        print("Processing %s" % dim)
        for m, expt_settings['method'] in enumerate(dims_methods): 
            print("Processing method %s" % expt_settings['method'])
    
            if m == 3 and (n == 1 or n==2):
                foldername_tmp = 'crowdsourcing_argumentation_expts_first_submission/'
                expt_settings_tmp = dict(expt_settings)
                expt_settings_tmp['feature_type'] = 'embeddings'
                expt_settings_tmp['foldorderfile'] = None
                if n == 1:
                    
                    expt_settings_tmp['embeddings_type'] = 'siamese-cbow'
                elif n == 2:
                    expt_settings_tmp['embeddings_type'] = 'skipthoughts'
                
                data, nFolds, resultsdir, resultsfile = load_results_data(data_root_dir, 
                                          resultsfile_template, expt_settings_tmp, foldername_tmp)
                print('***********')
                print(resultsfile)
                
                expt_settings_master = expt_settings
                expt_settings = expt_settings_tmp
            else:                                        
                if not len(dim):
                    continue
                data, nFolds, resultsdir, resultsfile = load_results_data(data_root_dir, 
                                          resultsfile_template, expt_settings, foldername)
                expt_settings_master = expt_settings
            runtimes_m = np.zeros(nFolds)
            print(resultsdir)
            for f in range(nFolds):
                print("Processing fold %i" % f)
                if expt_settings['fold_order'] is None: # fall back to the order on the current machine
                    if expt_settings['folds'] is None:
                        print("Skipping fold %i because no fold order file" % f)
                        continue
                    fold = list(expt_settings['folds'].keys())[f]
                else:
                    fold = expt_settings['fold_order'][f] 
                    if fold[-2] == "'" and fold[0] == "'":
                        fold = fold[1:-2]
                    elif fold[-1] == "'" and fold[0] == "'":
                        fold = fold[1:-1]  
                    expt_settings['fold_order'][f] = fold
                                              
                # look for new-style data in separate files for each fold. Prefer new-style if both are found.
                foldfile = resultsdir + '/fold%i.pkl' % f
                if os.path.isfile(foldfile):
                    with open(foldfile, 'rb') as fh:
                        data_f = pickle.load(fh, encoding='latin1')
                else: # convert the old stuff to new stuff
                    if data is None:
                        min_folds = f+1
                        print('Skipping fold with no data %i' % f)
                        print("Skipping results for %s, %s, %s, %s" % (expt_settings['method'], 
                                                                       expt_settings['dataset'], 
                                                                       expt_settings['feature_type'], 
                                                                       expt_settings['embeddings_type']))
                        print("Skipped filename was: %s, old-style results file would be %s" % (foldfile, 
                                                                                                resultsfile))
                        continue
                    
                    if not os.path.isdir(resultsdir):
                        os.mkdir(resultsdir)
                    data_f = []
                    for thing in data:
                        if f in thing:
                            data_f.append(thing[f])
                        else:
                            data_f.append(thing)
                    with open(foldfile, 'wb') as fh:
                        pickle.dump(data_f, fh)  
                                  
                runtimes_m[f] = data_f[6]
                
            if np.sum(runtimes_m>0):
                runtimes_dims[m, n] = np.mean(runtimes_m[runtimes_m>0])
                
            expt_settings = expt_settings_master     
    
    
    #gs = gridspec.GridSpec(2, 1, height_ratios=[0.5, 4])    
    fig4, ax4 = plt.subplots(figsize=(5,4))
    #ax4_2 = plt.subplot(gs[0])
    #ax4 = plt.subplot(gs[1], sharex=ax4_2)
    #fig4, (ax4_2, ax4) = plt.subplots(2, 1, sharex=True, figsize=(5,4))   
    
    x_dims = [1, 2, 3, 4.0322] # 3.5228353136605302, 
    #x_labels = [30, 300, 3000, 32310]
    x_ticklocs = [1, 2, 3,  4] # 3.5228353136605302,
    x_labels = ['3e1', '3e2', '3e3',  '3e4'] # '10e3',
    plt.xticks(x_ticklocs, x_labels)   
    
    runtimes_dims[1][0] = 397 # the observation in the dataset was wrong, use the times from the console printout
    runtimes_dims[1][2] = 2851
    
    h1, = ax4.plot(x_dims, runtimes_dims[1], label='SVM', marker='>', color='black', 
                   clip_on=False, linewidth=2, markersize=8)
    h2, = ax4.plot(x_dims, runtimes_dims[2], label='BiLSTM', marker='^', color='red', 
                   clip_on=False, linewidth=2, markersize=8)
    # this is too big
    vals = runtimes_dims[3][1:]
    print(vals)
    #vals = vals - 9800 + 3500 + 200
    h3, = ax4.plot(x_dims[1:], vals, marker='s', color='goldenrod', 
                   label='GPPL opt.', clip_on=False, linewidth=2, markersize=8)
    #plt.ylabel('Runtime (100s)')# for SVM/BiLSTM/GPPL, M=500, opt.')
    #plt.yticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 3700, 4200], 
    #           ['0', '500', '1000', '1500', '2000', '2500', '3000', '3500', '9800', '10300'])
#     ax4.set_ylim(0, 3500)
#     ax4_2.set_ylim(9500, 10000)
#     ax4_2.yaxis.set_ticks([9500,10000])
#     ax4_2.yaxis.set_ticklabels(['95', '100'])
#     ax4.yaxis.set_ticklabels(['0', '5', '10', '15', '20', '25', '30', '35'])
    plt.xlim(0.9, 4.1)
    plt.xlabel('No. Features')
    
    plt.tight_layout()
#     gs.update(hspace=0.07)
    
#     ax4_2.spines['bottom'].set_visible(False)
#     ax4.spines['top'].set_visible(False)  
#     ax4_2.xaxis.tick_top()
#     ax4_2.tick_params(labeltop='off')  # don't put tick labels at the top
#     ax4.xaxis.tick_bottom()
#     
#     d = .015  # how big to make the diagonal lines in axes coordinates
#     # arguments to pass to plot, just so we don't keep repeating them
#     kwargs = dict(transform=ax4_2.transAxes, color='k', clip_on=False)
#     ax4_2.plot((-d, +d), (0, 0), **kwargs)        # top-left diagonal
#     ax4_2.plot((1 - d, 1 + d), (0, 0), **kwargs)  # top-right diagonal
#     
#     kwargs.update(transform=ax4.transAxes)  # switch to the bottom axes
#     ax4.plot((-d, +d), (1, 1), **kwargs)  # bottom-left diagonal
#     ax4.plot((1 - d, 1 + d), (1, 1), **kwargs)  # bottom-right diagonal    
    
    ax4.legend(handles=[h3, h1, h2], labels=['GPPL opt.', 'SVM', 'BiLSTM'], 
                loc=(0.2, 0.6))    
    ax4.yaxis.grid('on')
    plt.savefig(figure_save_path + '/num_features_others.pdf')
    
    print("Runtimes for varying features GPPL medi: ") 
    print(runtimes_dims[0])
    
    fig5, ax5 = plt.subplots(figsize=(5,4))
    h4, = plt.plot(x_dims, runtimes_dims[0], label='GPPL medi.', marker='o', color='blue', 
                     clip_on=False, linewidth=2, markersize=8)
    plt.legend(loc='best')
    plt.xlabel('No. Features')
    ax5.yaxis.grid('on')
    plt.ylabel('Runtime (s)')
    #plt.ylabel('Runtime (s) for GPPL, M=500, medi.')
    plt.xticks(x_ticklocs, x_labels)   

    plt.savefig(figure_save_path + '/num_features_gppl.pdf')    