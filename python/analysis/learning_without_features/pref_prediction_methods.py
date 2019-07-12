'''
Library of various methods for predicting preferences using similarities between people (collaborative filtering). 
Uses clustering, factor analysis etc. to analyse the individuals and their preferences. Then uses various methods to
uses the clusters/factors to produce predictions for held-out data.

Created on 21 Oct 2016

@author: simpson
'''
from collab_pref_learning_vb import CollabPrefLearningVB
from gp_pref_learning import GPPrefLearning
from gp_classifier_vb import coord_arr_to_1d#, coord_arr_from_1d
import numpy as np
from sklearn.cluster.hierarchical import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.mixture import BayesianGaussianMixture#DPGMM GaussianMixture
from sklearn.decomposition import FactorAnalysis, LatentDirichletAllocation
import pickle, os, logging
#from scipy.optimize import fmin

class PredictionTester(object):
    
    def __init__(self, datadir, exptlabel, k, nx, ny, personids, pair1coords, pair2coords, prefs, train, 
                        test, results, rank_results, nfactors_max, nfactors_min):
        self.datadir = datadir
        self.exptlabel = exptlabel
        self.k = k
        self.nx = nx
        self.ny = ny
        self.personids = personids
        self.pair1coords = pair1coords
        self.pair2coords = pair2coords
        self.prefs = prefs
        self.trainidxs = train
        self.testidxs = test
        self.results = results 
        self.rank_results = rank_results
        self.nfactors_max = nfactors_max
        self.nfactors_min = nfactors_min
    
        # turn the data into a big matrix
        pair1coords_1d = coord_arr_to_1d(pair1coords)
        pair2coords_1d = coord_arr_to_1d(pair2coords)
                
        ucoords_1d, pairidxs_1d = np.unique([pair1coords_1d, pair2coords_1d], return_inverse=True)
        ncoords = len(ucoords_1d)
        self.pair1idxs = pairidxs_1d[:len(pair1coords_1d)]
        self.pair2idxs = pairidxs_1d[len(pair1coords_1d):]
        
#         # this may not be necessary as we could handle flipped comparisons in the method implementations
#         flipidxs = self.pair1idxs > self.pair2idxs
#         tmp = self.pair1idxs[flipidxs]
#         self.pair1idxs[flipidxs] = self.pair2idxs[flipidxs]
#         self.pair2idxs[flipidxs] = tmp
#         self.prefs[flipidxs] = 1 - self.prefs[flipidxs]       
        
        self.pairidxs_ravelled = np.ravel_multi_index((self.pair1idxs, self.pair2idxs), dims=(ncoords, ncoords))
        _, self.pairidxs_ravelled = np.unique(self.pairidxs_ravelled, return_inverse=True)
        self.npairs = np.max(self.pairidxs_ravelled) + 1
        
        self.nworkers = np.max(self.personids) + 1
        self.preftable = np.zeros((self.nworkers, self.npairs))
        self.preftable[:] = np.nan # + 0.5 # 0.5 is the default value
        self.preftable[self.personids, self.pairidxs_ravelled] = self.prefs
    
        self.preftable_train = np.zeros((self.nworkers, self.npairs)) + 0.5
        self.preftable_train[self.personids[self.trainidxs], self.pairidxs_ravelled[self.trainidxs]] = self.prefs[self.trainidxs]

        self.preftable_test = np.zeros((self.nworkers, self.npairs)) + 0.5
        self.preftable_test[self.personids[self.testidxs], self.pairidxs_ravelled[self.testidxs]] = self.prefs[self.testidxs]
        
        self.A = [] # affinity matrix -- don't compute until we need it
        
        self.most_common = np.nan
    
    def compute_affinity_matrix(self):
        filename = self.datadir + '/affinity_%s.pkl' % self.exptlabel
        if os.path.isfile(filename):
            with open(filename, 'r') as fh:
                self.A = pickle.load(fh)
                return
        
        # create an affinity matrix for clustering the raw data using the TRAINING DATA
        A = np.zeros((self.nworkers, self.nworkers))
        for i in range(self.nworkers):
            logging.debug('Creating affinity matrix, %i of %i rows' % (i, self.nworkers))
            agreement_i = self.preftable_train==self.preftable_train[i:i+1, :]
            A[i] = np.sum(agreement_i, axis=1) - np.sum(np.invert(agreement_i), axis=1)
        self.A = A
        with open(filename, 'w') as fh:
            pickle.dump(self.A, fh)        
    
    # Baselines
    def get_most_common_label(self):
        if not np.isnan(self.most_common):
            return self.most_common
        
        most_common = 0
        if np.sum(self.prefs==0) < np.sum(self.prefs==0.5):
            most_common = 0.5
            if np.sum(self.prefs==0.5) < np.sum(self.prefs==1):
                most_common = 1
        elif np.sum(self.prefs==0) < np.sum(self.prefs==1):
            most_common = 1
        self.most_common = most_common
        return self.most_common
    
    def run_baseline_most_common(self, m): # stupid baseline -- assigns the same label to all data points
        most_common = self.get_most_common_label()
        self.results[self.testidxs, m] = most_common
        
    def run_combine_all(self, m, gp=False):
        labels = np.zeros(self.nworkers) # they all belong to one cluster -- assume they are the same
        if gp:
            self.run_cluster_matching(labels, m)
        else:
            self.run_gp_per_cluster(labels, m)
        
    def run_affprop(self, m, gp_per_cluster=False):
        afprop = AffinityPropagation(affinity='precomputed')
        if not len(self.A):
            self.compute_affinity_matrix()
        
        labels =  afprop.fit_predict(self.A)
        
        if gp_per_cluster:
            self.run_gp_per_cluster(labels, m)
        else:
            self.run_cluster_matching(labels, m)

    def run_agglomerative(self, m, gp_per_cluster=False):
        agg = AgglomerativeClustering()
        labels = agg.fit_predict(self.preftable_train)
        if gp_per_cluster:
            self.run_gp_per_cluster(labels, m)
        else:
            self.run_cluster_matching(labels, m)
            
    def run_lda(self, m, gp_per_cluster=False):
        
        def neg_log_likelihood(ntopics, preftable_train):
            lda = LatentDirichletAllocation(n_topics=ntopics, learning_method='batch') # need to optimise ncomponentns using lda.score()
            lda.fit(preftable_train)
            score = - lda.score(preftable_train)
            return score, lda

        minscore = np.inf        
        for ntopics in range(self.nfactors_min, self.nfactors_max+1):
            logging.info('Trying ntopics=%i' % ntopics)            
            score, lda_n = neg_log_likelihood(ntopics, self.preftable_train)
            if score < minscore:
                logging.info('Choosing ntopics=%i' % ntopics)
                minscore = score
                lda = lda_n
        
        workertopics = lda.transform(self.preftable_train)
        if gp_per_cluster:
            logging.error('Not implemented yet: LDA with GPs trained on each cluster. ') 
            # Problem is that the soft cluster membership means we don't have a definite set of data points to train
            # a GP for each cluster. To resolve this, we need to understand how we can use the cluster probabilities
            # to adapt weaken the observations for training the GP. 
            #self.run_soft_gp_per_cluster(workertopics, m)
        else:
            self.run_soft_cluster_matching(workertopics, m)
            
    def run_raw_gmm(self, m, gp_per_cluster=False, soft_cluster_matching=False):
        
        gmm = BayesianGaussianMixture(n_components=self.nfactors_max, weight_concentration_prior=0.5, 
                                      weight_concentration_prior_type='dirichlet_process', covariance_type='diag')
        gmm.fit(self.preftable_train)
        labels = gmm.predict(self.preftable_train)
        if gp_per_cluster:
            self.run_gp_per_cluster(labels, m)
        elif soft_cluster_matching:
            weights = gmm.predict_proba(self.preftable_train)
            self.run_soft_cluster_matching(weights, m)
        else:
            self.run_cluster_matching(labels, m)
                    
    def run_gp_affprop(self, m, gp_per_cluster=False):
        fbar = self.run_gp_separate(m)
        
        afprop = AffinityPropagation()        
        labels = afprop.fit_predict(fbar)
        if gp_per_cluster:
            self.run_gp_per_cluster(labels, m)
        else:
            self.run_cluster_matching(labels, m)
                      
    # gmm on the separate fbars  
    def run_gp_gmm(self, m, gp_per_cluster=False, soft_cluster_matching=False):
        fbar = self.run_gp_separate(m)

        gmm = BayesianGaussianMixture(n_components=self.nfactors_max, weight_concentration_prior=0.1, 
                                      weight_concentration_prior_type='dirichlet_process', covariance_type='diag') 
        gmm.fit(fbar)
        labels = gmm.predict(fbar)
        if gp_per_cluster:
            self.run_gp_per_cluster(labels, m)
        elif soft_cluster_matching:
            weights = gmm.predict_proba(fbar)
            self.run_soft_cluster_matching(weights, m)
        else:
            self.run_cluster_matching(labels, m)
            
    def run_fa(self, m):
        fbar = self.run_gp_separate(m)
        
        def neg_log_likelihood(nfactors, fbar):
            fa = FactorAnalysis(nfactors)
            fa.fit(fbar)
            score = - fa.score(fbar)
            return score, fa

        minscore = np.inf        
        for nfactors in range(self.nfactors_min, self.nfactors_max+1):
            logging.info('Trying nfactors=%i' % nfactors)            
            score, fa_n = neg_log_likelihood(nfactors, fbar)
            if score < minscore:
                logging.info('Choosing nfactors=%i' % nfactors)
                minscore = score        
                fa = fa_n        
        
        y = fa.transform(fbar)
        
        self.run_fa_matching(y, m)
            
    def fit_predict_gp(self, pair1coords_train, pair2coords_train, prefs, pair1coords_test, pair2coords_test, 
                       return_latent_f=False):
        # TODO: update the parameters in function calls
        model = GPPrefLearning([self.nx, self.ny], mu0=0,shape_s0=1, rate_s0=1, ls_initial=[10, 10])
        model._select_covariance_function('diagonal')
        model.max_iter_VB = 50
        model.min_iter_VB = 10
        model.max_iter_G = 3      
        model.verbose = False
        model.uselowerbound = False

        logging.info('Fitting GP...')
        model.fit(pair1coords_train, pair2coords_train, prefs) # ignores any user ids
        logging.info('Fitted. Predicting from GP...')
        # does model.f cover all the data points? If not, we should be able to pass that in
        results, _ = model.predict_pairs_from_features(pair1coords_test, pair2coords_test)
        logging.info('Predicted.')
        if return_latent_f:
            return results.flatten(), model.f.flatten()
        else:
            return results.flatten()
            
    def run_gp_per_cluster(self, labels, m):
       
        #get the clusters of the personids
        clusters_test = labels[self.personids[self.testidxs]]
        clusters_train = labels[self.personids[self.trainidxs]]     
        
        uclusters = np.unique(labels)
        for cl in uclusters:
            clidxs = clusters_train==cl
            clidxs_test = clusters_test==cl
            logging.debug("--- Running GP pref model for cluster %i ---" % cl)
            if not np.sum(clidxs_test) or not np.sum(clidxs):
                continue
            results = self.fit_predict_gp(self.pair1coords[self.trainidxs][clidxs], 
                                          self.pair2coords[self.trainidxs][clidxs], 
                                          self.prefs[self.trainidxs][clidxs], 
                                          self.pair1coords[self.testidxs][clidxs_test], 
                                          self.pair2coords[self.testidxs][clidxs_test])
        
            self.results[self.testidxs[clidxs_test], m] = results
            
        # find the results that are still at 0.5
        notlabelledidxs = self.results[self.testidxs, m] == 0.5
        
        logging.debug("--- Running Pref GP model for all workers --- ")
        # TODO: do the results of this actually get saved anywhere? 
        self.fit_predict_gp(self.pair1coords[self.trainidxs], 
                            self.pair2coords[self.trainidxs], 
                            self.prefs[self.trainidxs], 
                            self.pair1coords[self.testidxs][notlabelledidxs], 
                            self.pair2coords[self.testidxs][notlabelledidxs])

    def run_cluster_matching(self, labels, m):
       
        #get the clusters of the personids
        clusters_test = labels[self.personids[self.testidxs]]
        clusters_train = labels[self.personids[self.trainidxs]]
        
        prob_pref_test = np.zeros(self.testidxs.size) # container for the test results
        
        most_common_label = np.nan
        
        #get the other members of the clusters, then get their labels for the same pairs
        for i, cl in enumerate(clusters_test):
            members = clusters_train == cl #pairs from same cluster
            pair1 = self.pair1idxs[self.testidxs[i]] # id for this current pair
            pair2 = self.pair2idxs[self.testidxs[i]]
            # idxs for the matching pairs 
            matching_pair_idxs = ((self.pair1idxs[self.trainidxs]==pair1) & (self.pair2idxs[self.trainidxs]==pair2))
            # total preferences for the matching pairs 
            nannotators_for_this_pair = np.sum(members)
            cluster_size = np.sum(matching_pair_idxs & members) + 1.0
            
            if cluster_size > 1.0:
                total_prefs_matching = np.sum((self.prefs[self.trainidxs][matching_pair_idxs & members] - 0.5) * 2)                
                prob_pref_test[i] = (float(total_prefs_matching) / float(cluster_size) + 1) / 2.0
            elif not np.sum(matching_pair_idxs): # no other have labelled this pair
                if np.isnan(most_common_label):
                    most_common_label = self.get_most_common_label()
                prob_pref_test[i] = most_common_label
            else: # others have labelled this pair, but not in same cluster
                #prob_pref_test[i] = self.get_most_common_label() # use most common label
                # take an average of all the workers
                total_prefs_matching = np.sum(self.prefs[self.trainidxs][matching_pair_idxs])
                cluster_size = nannotators_for_this_pair + 1.0
                prob_pref_test[i] = float(total_prefs_matching) / float(cluster_size)
                
        self.results[self.testidxs, m] = prob_pref_test

    def run_soft_cluster_matching(self, weights, m):       
        #get the clusters of the personids
        prob_pref_test = np.zeros(self.testidxs.size) # container for the test results
                
        most_common_label = np.nan

        #get the other members of the clusters, then get their labels for the same pairs
        for i, idx in enumerate(self.testidxs):
            pair1 = self.pair1idxs[idx] # id for this current pair
            pair2 = self.pair2idxs[idx]
            # idxs for the matching pairs 
            matching_pair_idxs = ((self.pair1idxs[self.trainidxs]==pair1) & (self.pair2idxs[self.trainidxs]==pair2))
            if not np.sum(matching_pair_idxs): # no others have labelled this pair
                if np.isnan(most_common_label):
                    most_common_label = self.get_most_common_label()
                prob_pref_test[i] = most_common_label
                
            # total preferences for the matching pairs 
            p = self.personids[idx]
            matching_people = self.personids[self.trainidxs][matching_pair_idxs]
            weighted_matches = weights[matching_people, :] * weights[p:p+1, :]
            weighted_matches = np.sum(weighted_matches, axis=1)[:, np.newaxis]
            cluster_size = np.sum(weighted_matches) + 1.0
            weighted_matches /= cluster_size
            
            prob_pref_test[i] = np.sum(self.prefs[self.trainidxs][matching_pair_idxs][:, np.newaxis] * weighted_matches)
                
        self.results[self.testidxs, m] = prob_pref_test
        
    def run_fa_matching(self, factors, m):
        #get the clusters of the personids
        prob_pref_test = np.zeros(self.testidxs.size) # container for the test results
        
        #get the other members of the clusters, then get their labels for the same pairs
        for i, idx in enumerate(self.testidxs):
            pair1 = self.pair1idxs[idx] # id for this current pair
            pair2 = self.pair2idxs[idx]
            # idxs for the matching pairs 
            matching_pair_idxs = ((self.pair1idxs[self.trainidxs]==pair1) & (self.pair2idxs[self.trainidxs]==pair2))
            # total preferences for the matching pairs 
            # gaussian kernel -- pdf of the matching pairs given the current pair as the mean
            p = self.personids[idx]
            matching_people = self.personids[self.trainidxs][matching_pair_idxs]
            sqdist = 0.5 * (factors[p:p+1, :] - factors[matching_people, :])**2
            weighted_matches = np.sum(np.exp(-sqdist), axis=1)[:, np.newaxis] 
            cluster_size = np.sum(weighted_matches) + 1.0
            weighted_matches /= cluster_size
            
            prob_pref_test[i] = np.sum(self.prefs[self.trainidxs][matching_pair_idxs][:, np.newaxis] * weighted_matches)
                
        self.results[self.testidxs, m] = prob_pref_test
    
    def run_gpfa_bayes(self, m):
        # Task C1  ------------------------------------------------------------------------------------------------
        def neg_log_likelihood(nfactors):
            model_gpfa = CollabPrefLearningVB([self.nx, self.ny], mu0=0, shape_s0=1, rate_s0=1, ls_initial=[10, 10],
                                              verbose=False, nfactors=nfactors)
            #model_gpfa.verbose = False
            model_gpfa.cov_type = 'diagonal'
            model_gpfa.fit(self.personids[self.trainidxs], self.pair1coords[self.trainidxs], 
                       self.pair2coords[self.trainidxs], self.prefs[self.trainidxs])           
            score = - model_gpfa.lowerbound()
            return score, model_gpfa

        minscore = np.inf        
        for nfactors in range(self.nfactors_min, self.nfactors_max+1):
            logging.info('Trying nfactors=%i' % nfactors)
            score, fa_n = neg_log_likelihood(nfactors)
            if score < minscore:
                logging.info('Choosing nfactors=%i' % nfactors)
                minscore = score
                model_gpfa = fa_n          
        results_k = model_gpfa.predict(self.personids[self.testidxs], self.pair1coords[self.testidxs], 
                                       self.pair2coords[self.testidxs])
        self.results[self.testidxs, m] = results_k
        
        return results_k, model_gpfa
    
    def run_gp_separate(self, m):
        #run the model but without the FA part; no shared information between people. 
        #Hypothesis: splitting by person results in too little data per person

        try:
            return self.fbar
        except:
            logging.debug('Need to compute the separate GPs for each person.')
            
        upersonids = np.unique(self.personids)
        ncoords = np.unique([coord_arr_to_1d(self.pair1coords), coord_arr_to_1d(self.pair2coords)]).size
        self.fbar = np.zeros((self.nworkers, ncoords))

        for p in upersonids:   
            pidxs_train = self.trainidxs[self.personids[self.trainidxs]==p]
            if not len(pidxs_train):
                continue
            logging.info('Training input GP for person %i' % p)
            _, self.fbar[p, :] = self.fit_predict_gp(
                                            self.pair1coords[pidxs_train], 
                                            self.pair2coords[pidxs_train], 
                                            self.prefs[pidxs_train], 
                                            self.pair1coords, # test on all the available data 
                                            self.pair2coords,
                                            return_latent_f=True)
        return self.fbar