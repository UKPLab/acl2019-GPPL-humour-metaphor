'''

For the list of tasks implemented here, see:

https://docs.google.com/spreadsheets/d/15LXSrCcaDURsIYmogt-NEwaakUiITFtiq993TLkflRQ/edit#gid=0

Created on 10 May 2016

@author: simpson
'''
import logging, os, datetime, pickle
logging.basicConfig(level=logging.DEBUG)
import numpy as np
from mpl_toolkits.mplot3d import Axes3D #@UnresolvedImport  
from sklearn.cross_validation import KFold


import classification_metrics
from pref_prediction_methods import PredictionTester
from preproc_raw_data import load_amt, load_synthetic, exptlabel

if __name__ == '__main__':
    # Experiment configuration ----------------------------------------------------------------------------------------

#     dataset = 'amt'
#     split_results_by_no_annotations = True
#     nruns = 1
    
    synth_worker_accs = [0.7, 0.8, 0.9] # [0.9]    
    dataset = 'synth'
    nruns = len(synth_worker_accs)
    split_results_by_no_annotations = False

    # number of factors or components that we can try in our models
    nfactors_min = 5
    nfactors_max = 5
    
    # list the names of methods to test here
    methods = [
            # seems like we are missing alternative learning-to-rank methods -- only have GP.
            'Combined_GP',
            # Noise: current approach is to de-noise with MACE, then feed to other classifiers. Rankings are then 
            # obtained using PageRank, which could also be used to produce pairwise classifications if no feature data.
            # Testing with feature data would show whether features actually help to de-noise, plus allow comparison
            # between GP and MACE+SVM at different noise levels. 
            
            # Sparsity: again, we want to compare the initial fusion step using MACE with the GP. Similar comparisons
            # needed as in previous step.
            
            # A single plot (set of subplots)? We can explore noise and sparsity using a single set of plots so that 
            # we can show the interplay between noise and sparsity. Make separate sets of plots for CEE and ACC.
            # Each subplot shows the variation in performance as we introduce more pairs at a given noise level.
            # Three different noise levels should be sufficient.
            
            # Step 1: get the setup running with the post-MACE dataset (handling the noise in crowdsourced data is 
            # another topic; however, our method should still deal with the noise/uncertainty that comes out of MACE).
            # Step 2: introduce the sparsity variation.
            # Step 3: loop over different noise levels.
            # Step 4: plot
            # Step 5: page rank (can output pair classifications by looking at rankings).
            # Step 6: SVR (trained on pagerank) and SVC methods.
            # Step 7: Update and save plots.
            
            
            #'Baseline_MostCommon', 
            #'Combined_Averaging', # these two only make sense with multiple labels from different workers?
            # personalised, factored and clustered models
#             'GPFABayes_GP',
#             'Separate_GP',  
#             'GPFA_soft',  # no. factors?
#             'GPGMM_GP', 
#             'GPGMM_soft', 
#             'GPGMM_Averaging',
#             'GPAffProp_GP', # this takes ages because there are too many clusters
#             'AffProp_Averaging', 
#             'AffProp_GP', 
#             'Agg_Averaging', 
#             'Agg_GP', 
#             'LDA_soft' # no. topics? Needs posterior over no. topics, not likelihood
            ] 
            #'GMM_GP', 'GMM_Averaging', 'GMM_soft'] # these don't make any sense as cannot learn GMM from a discrete data
            #'GPGMM_GP', 'GPGMM_soft', 'Separate_GP', 'GPAffProp_GP' # this may be infeasible with lots of workers 
    nmethods = len(methods) 
    #2 * len(nfactors_list) + 1 # need to increase nmethods if we are trying multiple numbers of factors 
    # -- new implementation will try to optimize the number of factors internally and return only the best results for each method

    # RUN THE VARIOUS PREFERENCE PREDICTION METHODS....
    k = 0 # count which fold we are in so we can save data    

    nfolds = 10

    metrics = {} # classification/pairwise preference performance metrics
    rank_metrics = {} #ordering of items 
    
    # Task A2 ---------------------------------------------------------------------------------------------------------
    for r in range(nruns): # Repeat everything!
        # comment out down to the beginning of the for loop if we need to restart the process from fold number 6
        if dataset == 'synth':
            datadir, plotdir, nx, ny, data, pair1coords, pair2coords, pair1idxs, pair2idxs, xvals, yvals, prefs, personids,\
                npairs, nworkers, ntexts, f = load_synthetic(synth_worker_accs[r]) # use different params in each run
            xlabels = synth_worker_accs
            # make sure the root exists
            if not os.path.isdir(plotdir):
                os.mkdir(plotdir)
            plotdir += '/synth/'
            # make sure the subdirectory exists
            if not os.path.isdir(plotdir):
                os.mkdir(plotdir)
        else:
            datadir, plotdir, nx, ny, data, pair1coords, pair2coords, pair1idxs, pair2idxs, xvals, yvals, prefs, \
                personids, npairs, nworkers, ntexts, f = load_amt()
            # make sure the root exists    
            if not os.path.isdir(plotdir):
                os.mkdir(plotdir)                
            plotdir += '/amt/'
            # make sure the subdirectory exists            
            if not os.path.isdir(plotdir):
                os.mkdir(plotdir)            
            xlabels = None
        if not os.path.isdir(plotdir):
            os.mkdir(plotdir)
         
        kf = KFold(npairs, nfolds, shuffle=True, random_state=100)
           
        results = np.zeros((npairs, nmethods)) # the test results only
        rank_results = np.zeros((ntexts, nmethods)) # used by methods that rank items as well as predict pairwise labels
            
        for train, test in kf:
            logging.info('Running fold %i' % k)
                        
            tester = PredictionTester(datadir, exptlabel, k, nx, ny, personids, pair1coords, pair2coords, prefs, train, test, results, 
                                      rank_results, nfactors_max, nfactors_min)
            
            for m, method in enumerate(methods):
                
#                 # restart from fold 6 for all methods except those specified below
#                 if k < 6 and method != 'GPFA_soft' and method != 'LDA_soft' and method != 'GPGMM_soft':
#                     k += 1
#                     continue                
                
                start = datetime.datetime.now()
                
                clustermethod = method.split('_')[0]
                predictionmethod = method.split('_')[1]
                
                if predictionmethod=='GP':
                    gppredictionmethod = True
                    softprediction = False
                    predictionmethod_str = 'learning a GP for each cluster to predict preferences'
                elif predictionmethod=='soft':
                    gppredictionmethod = False
                    softprediction = True
                    predictionmethod_str = 'averaging clusters using soft membership weights to predict'
                else:
                    gppredictionmethod = False
                    softprediction = False
                    predictionmethod_str = 'averaging clusters to predict'
                
                # baseline: assign most common class label
                if method=='Baseline_MostCommon':
                    logging.info('Baseline -- Assign most common class')
                    tester.run_baseline_most_common(m)
                    
                # Task C3: Baseline with no separate preference functions per user or clustering 
                elif clustermethod=='Combined':
                    logging.info('Treat all workers as same and average')
                    tester.run_combine_all(m, gp=gppredictionmethod)
                
                # clustering the raw preferences
                elif clustermethod == 'AffProp':
                    logging.info('Affinity Propagation, then %s' % predictionmethod_str)
                    
                    tester.run_affprop(m, gp_per_cluster=gppredictionmethod)
                elif clustermethod == 'Agg':
                    logging.info('Agglomerative clustering, then %s' % predictionmethod_str)
                    
                    tester.run_agglomerative(m, gp_per_cluster=gppredictionmethod)
                elif clustermethod == 'GMM':
                    logging.info('Gaussian mixture, then %s' % predictionmethod_str)
                    
                    tester.run_raw_gmm(m, gp_per_cluster=gppredictionmethod, soft_cluster_matching=softprediction)
                elif clustermethod == 'LDA':
                    logging.info('LDA, then %s' % predictionmethod_str)
                    tester.run_lda(m, gp_per_cluster=gppredictionmethod) 
                    
                # testing whether the smoothed, continuous GP improves clustering
                # the effect may only be significant once we have argument features
                # assuming no clustering at all, but using a GP to smooth 
                elif method=='Separate_GP':              
                    logging.info('Task C1 part II, Separate GPs (no FA)')
                    tester.run_gp_separate(m)
                    
                elif clustermethod=='GPFA':
                    logging.info('Task C1, GPFA, separate steps with distance-based matching')
                    tester.run_fa(m)
                
                # factor analysis + GP         
                elif clustermethod=='GPFABayes':
                    logging.info('Task C1, GPFA, Bayesian method')
                    # Run the GPFA with this fold
                    tester.run_gpfa_bayes(m)   
                                 
                # clustering methods on top of the smoothed functions
                elif clustermethod=='GPAffProp':
                    logging.info('Preference GP, function means fed to Aff. Prop., then %s' % predictionmethod_str)
                    tester.run_gp_affprop(m, gp_per_cluster=gppredictionmethod)
                    
                elif clustermethod=='GPGMM':
                    logging.info('Preference GP, function means fed to GMM, then %s' % predictionmethod_str)
                    tester.run_gp_gmm(m, gp_per_cluster=gppredictionmethod, 
                                      soft_cluster_matching=softprediction)
                                    
                end = datetime.datetime.now()
                duration = (end - start).total_seconds()
                logging.info("Method %i in fold %i took %.2f seconds" % (m, k, duration))
                               
            k += 1
              
        if split_results_by_no_annotations:
            # separate the data points according to the number of annotators    
            paircounts = np.sum(np.invert(np.isnan(tester.preftable)), axis=0)
            nanno_max = 10
            for nanno in range(2, nanno_max+1):
                if nanno==nanno_max:
                    pairs = np.argwhere(paircounts >= nanno)
                else:
                    pairs = np.argwhere(paircounts == nanno)
                if not pairs.size:
                    continue
                #ravel the pair1 and pair2 idxs
                pairidxs = np.in1d(tester.pairidxs_ravelled, pairs)
                gold = prefs[pairidxs]
                if len(np.unique(gold)) != 3:
                    # we don't have examples of all classes
                    continue
                
                res = results[pairidxs, :]
                
                metrics = classification_metrics.compute_metrics(nmethods, gold, res, metrics, nruns, r)
                if r == nruns - 1:
                    if nanno == nanno_max:  
                        classification_metrics.plot_metrics(plotdir, metrics, nmethods, methods, nfolds, nanno, 
                                                            nanno_is_min=True, xlabels=xlabels)
                    else:
                        classification_metrics.plot_metrics(plotdir, metrics, nmethods, methods, nfolds, nanno, 
                                                            xlabels=xlabels)
        
                if np.any(rank_results):
                    rank_metrics = classification_metrics.compute_ranking_metrics(nmethods, f, rank_results, 
                                                                                  rank_metrics, nruns, r)
                    if r == nruns - 1:
                        if nanno == nanno_max:  
                            classification_metrics.plot_metrics(plotdir, rank_metrics, nmethods, methods, nfolds, 
                                                                nanno, nanno_is_min=True, xlabels=xlabels)
                        else:
                            classification_metrics.plot_metrics(plotdir, rank_metrics, nmethods, methods, nfolds, 
                                                                nanno, xlabels=xlabels)
                        
        metrics = classification_metrics.compute_metrics(nmethods, prefs, results, metrics, nruns, r)
        
        with open(plotdir + "/metrics.pkl", 'w') as fh:
            pickle.dump(metrics, fh)
        
        if r == nruns - 1:    
            classification_metrics.plot_metrics(plotdir, metrics, nmethods, methods, nfolds, 1, nanno_is_min=True, 
                                                xlabels=xlabels)
        
        if np.any(rank_results):
            rank_metrics = classification_metrics.compute_ranking_metrics(nmethods, f, rank_results, rank_metrics, 
                                                                          nruns, r)
            if r == nruns - 1:
                classification_metrics.plot_metrics(plotdir, rank_metrics, nmethods, methods, nfolds, nanno, 
                                                nanno_is_min=True, xlabels=xlabels)