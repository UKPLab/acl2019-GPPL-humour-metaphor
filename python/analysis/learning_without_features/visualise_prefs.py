'''
Code to plot the raw preferences and the latent functions inferred using GP preference learning on each individual. 

Created on 21 Oct 2016

@author: simpson
'''
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
import pickle, logging
from krippendorffalpha import alpha
from scipy.stats import norm
from preproc_raw_data import load
from pref_prediction_methods import PredictionTester
from gp_classifier_vb import coord_arr_to_1d
from scipy.sparse import coo_matrix
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture#DPGMM
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import FactorAnalysis
import os

# Visualise the raw preference label data -----------------------------------------------------------------------------
    
def plot_raw_prefs(plotdir, data, xvals, pair1idxs, pair2idxs, N):
    # Show how much variation there is in the preference labels awarded to each pair.
    # Histogram of the number of preferences of each type received for each pair. Scatter plot showing the pref labels
    # received for each pair. Plot the variance in the preference labels given to each pair.
         
         
    # B6 plot histograms of the gold standard preference pairs. Plots to try:
    # x-axis indexes the argument pairs
    # y-axis indexes the number of observations. 
    # sort by score: number of positive - negative labels
    # Alternatively, summarise this plot somehow?
    logging.info('B6 start')
     
    plt.figure()
     
    p_hist = np.zeros((3, N**2 ))
    p_scat_x = np.zeros((data.shape[0]))
    p_scat_y = np.zeros((data.shape[0]))
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            pairidxs = ((xvals[pair1idxs]==i) & (xvals[pair2idxs]==j)) | ((xvals[pair2idxs]==i) & (xvals[pair1idxs]==j))
            p_hist[0, idx] = np.sum(data[pairidxs, 3] == 0)
            p_hist[1, idx] = np.sum(data[pairidxs, 3] == 1)
            p_hist[2, idx] = np.sum(data[pairidxs, 3] == 2)
             
            if np.any(pairidxs):
                p_scat_x[pairidxs] = idx
                p_scat_y[pairidxs] = data[pairidxs, 3]
             
    logging.info('B6 loop complete')
    # sort by mean value
    means = np.sum(p_hist * [[-1], [0], [1]], axis=0)
    sortbymeans = np.argsort(means)
    p_hist = p_hist[:, sortbymeans]
             
    # x locations
    x_locs = np.arange(N**2) - 0.5
     
    #plot histogram
    width = 0.3
    plt.bar(x_locs, p_hist[0, :], width, label='1 > 2')
    plt.bar(x_locs + width, p_hist[1, :], width, label='1==2')
    plt.bar(x_locs + 2*width, p_hist[2, :], width, label='1 < 2')
     
    plt.xlabel('Argument Pairs')
    plt.ylabel('Number of labels')
    plt.legend(loc='best')
    plt.title('Histogram of Labels for each Argument')
     
    plt.savefig(plotdir + '/b6_pref_histogram.eps')

    #scatter plot
    plt.scatter(p_scat_x, p_scat_y)
    plt.xlabel('Argument Pairs')
    plt.ylabel('Preference Label')
    plt.title('Distribution of Preferences for Arguments')
     
    plt.savefig(plotdir + '/b6_pref_scatter.eps')
    
    # B7 Compute variance in the observed preferences and sort
    mean_p_hist = (-1 * p_hist[0, :] + 1 * p_hist[2, :]) / np.sum(p_hist, axis=0)
    var_p_hist = (p_hist[0, :] - mean_p_hist)**2 + (p_hist[1, :] - mean_p_hist)**2 + (p_hist[2, :] - mean_p_hist)**2
    var_p_hist /= np.sum(p_hist, axis=0) 
     
    var_p_hist = np.sort(var_p_hist)
     
    # B8 Plot Preference pair variance and save
    plt.figure()
    plt.plot(x_locs + 0.5, var_p_hist)
    plt.xlabel('Argument Pairs')
    plt.ylabel('Variance in Pref. Labels')
    plt.title('Variance in Labels Collected for Each Pair')
     
    plt.savefig(plotdir + '/b8_pref_pair_var.eps')    

# Visualise Preference Functions inferred separately for each person --------------------------------------------------

def plot_pref_func_totals_all_people(plotdir, datadir, density_xvals, findividual):
    #Show how much the preference function varies between items across the whole population
    
    fsum = np.sum(findividual, axis=2)        
     
    #order the points by their midpoints (works for CDF?)
    #midpoints = fsum[density_xvals.shape[0]/2, :]
    peakidxs = np.argmax(fsum, axis=0)
    ordering = np.argsort(peakidxs)
    fsum = fsum[:, ordering]
 
    with open (datadir + '/b1_fsum.pkl', 'w') as fh:
        pickle.dump(fsum, fh)
    
    # B2. 3D plot of the distribution. 
 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
     
    # B3. Produce/save the plot
    idxmatrix = np.arange(fsum.shape[1])
    idxmatrix = np.tile(idxmatrix[np.newaxis, :], (density_xvals.shape[0], 1)) # matrix of xvalue indices
    ax.plot_surface(density_xvals, idxmatrix, fsum, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.savefig(plotdir + 'b3_fsum.eps')
    
def plot_pref_func_var_between_people(plotdir, datadir, fbar, seenidxs, ):
    # Show which items have the greatest disagreement
    
    # B4. Compute variance of the GP means and sort by variance
    fbar_seen = np.empty(fbar.shape) # need to deal properly with the nans
    fbar_seen[:, :] = np.NAN 
    fbar_seen[seenidxs] = fbar[seenidxs]
    fmean_var = np.nanvar(fbar_seen, axis=0) # should exclude the points where people made no classification
     
    fmean_var = np.sort(fmean_var)
     
    with open (datadir + '/fmean_var.pkl', 'w') as fh:
        pickle.dump(fmean_var, fh)
     
    # B5. Plot variance in pref function means
    plt.figure()
    plt.plot(np.arange(len(fmean_var)), fmean_var)
    plt.xlabel('Argument Index')
    plt.ylabel('Variance in Latent Pref Function Expected Values')
    plt.title('Variance in Expected Latent Preferences Between Different Members of the Crowd')
    plt.savefig(plotdir + 'b5_fsum.eps')
    
def plot_pref_func_means(plotdir, fbar, N):
    # Compare the preference functions of each person.
    
    # B9 Plot pref function means as a line graph -- without using a model, this will be very hard to read
    plt.plot(np.arange(N)[:, np.newaxis], fbar.T)
    plt.xlabel('Arguments')
    plt.ylabel('Latent Preference Value')
    plt.title('Expected Latent Preference Functions for Each Person')
     
    plt.savefig(plotdir + '/b9_pref_means.eps')
        
# Cluster Visualisation -----------------------------------------------------------------------------------------------

def bar_cluster_membership(plotdir, ncomponents, membership, label):
    # Bar chart
    
    plt.figure()
    _, ax = plt.subplots()    
    
    for m in range(len(membership)):
        plt.bar(np.arange(ncomponents), membership[m], label=label[m])
    ax.set_xticklabels(np.arange(ncomponents))
    plt.title('Total membership in Each Cluster')
    plt.xlabel('Cluster Index')
    plt.ylabel('Number of Members')
    plt.legend(loc='best')
     
    plt.savefig(plotdir + '/cluster_membership_%s.eps' % (label))

def plot_cluster_membership_probs(plotdir, nflabel, ncomponents, proba, label, nworkers):
    plt.figure()
    plt.title('Distribution of Cluster Membership Probabilities with %i Clusters' % ncomponents)
    for k in range(ncomponents):
        plt.subplot(ncomponents / 4 + 1, 4, k+1 )
        plt.scatter(np.arange(nworkers), proba[:, k])
        plt.xlim(-5 * nworkers, 6 * nworkers)
        plt.xlabel('Indexes of People')
        plt.ylabel('Mixture Component Weight')            
    plt.savefig(plotdir + '/cluster_probs_%s_%s.eps' % (nflabel, label))
    
def plot_cluster_pair_probs(plotdir, nflabel, ncomponents, proba, label):
    plt.figure()
    plt.title('Distribution of People between Pairs of Clusters (%i Clusters)' % ncomponents)
    for k in range(ncomponents):
        for k2 in range(ncomponents):
            plt.subplot(ncomponents**2 / 4 + 1, 4, k*ncomponents + k2 + 1 )
            plt.scatter(proba[:, k], proba[:, k2])
            plt.xlabel('probability of cluster %i' % k)
            plt.ylabel('probability of cluster %i' % k2)                
            plt.title('Factors %i and %i' % (k, k2))
    plt.savefig(plotdir + '/cluster_probs_pairs_%s_%s.eps' % (nflabel, label))    
        

def plot_pref_func_stats_by_cluster(plotdir, ncomponents, weights, label, density_xvals, findividual):
    plt.figure()
    plt.title('Latent Function mean and Variance by Cluster %s' % label)
        
#     scalarMap = cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('c', lut)
        
    for k in range(ncomponents):
        # use a set of samples from the mixture distribution to approximate the mean.
        if np.sum(weights[k]) < 2:
            continue
        
        fsum_k = findividual * weights[k] / np.sum(weights[k])
        fsum_k = np.sum(fsum_k, axis=2)
        #fsum_k = fsum_k[:, ordering]
    
        # Plot the means only
        fmean_k = np.sum(density_xvals *  fsum_k / np.sum(fsum_k, axis=0)[np.newaxis, :], axis=0)
        x = np.arange(fsum_k.shape[1])
        plt.plot(x, fmean_k, label='mean for cluster %s' % k)
        
        # compute the variance and put into the same plot
        fvar_k = np.sum((density_xvals - fmean_k[np.newaxis, :])**2 * fsum_k, axis=0) /  np.sum(fsum_k, axis=0)
        plt.fill_between(x, fmean_k + fvar_k, fmean_k + fvar_k, alpha=0.4, label='variance for cluster %s' % k)    
        
    plt.legend(loc='best')
    plt.xlabel('Arguments')
    plt.ylabel('Latent preference function')
    plt.savefig(plotdir + 'd5_fmeank_%s.eps' % (label))        
    
def plot_predicted_pref_stats_by_cluster(plotdir, ncomponents, weights, label, fbar, N):
    
    plt.figure()
    plt.title('Means and Variances of Predicted Preferences for Each Cluster %s' % label)
    
    for k in range(ncomponents):
        # use a set of samples from the mixture distribution to approximate the mean.
        if np.sum(weights[k]) < 2:
            continue
        prefs_mean = np.zeros(N**2)
        prefs_var = np.zeros(N**2)
        for i in range(N):
            for j in range(N):
                idx = i * N + j    
                prefs_ij =  (fbar[:, i] > fbar[:, j])          
                prefs_mean[idx] = prefs_ij * weights[k] / np.sum(weights[k])

                prefs_var[idx] = (prefs_ij - prefs_mean[idx])**2 * weights[k] / np.sum(weights[k])
        
        x = np.arange(N**2)
        plt.plot(x, prefs_mean, label='mean for cluster %s' % k)
        
        # compute the variance
        plt.fill_between(x, prefs_mean - prefs_var, prefs_mean + prefs_var, alpha=0.4, label='variance for cluster %s' % k)    
        
    plt.legend(loc='best')
    plt.xlabel('Arguments')
    plt.ylabel('Fraction of Preferences for a > b')
    plt.savefig(plotdir + 'd8_predprefs_%s.eps' % (label))    
    
def plot_raw_pref_stats_by_cluster(plotdir, ncomponents, weights, label, data, xvals, pair1idxs, pair2idxs, N):
    
    plt.figure()
    plt.title('Means and Variances of Observed Preferences for Each Cluster %s' % label)
    
    for k in range(ncomponents):
        # use a set of samples from the mixture distribution to approximate the mean.
        if np.sum(weights[k]) < 2:
            continue
        prefs_mean = np.zeros(N**2)
        prefs_var = np.zeros(N**2)
        for i in range(N):
            for j in range(N):
                idx = i * N + j
                pairidxs = (xvals[pair1idxs]==i) & (xvals[pair2idxs]==j)
                pair_rev_idxs = (xvals[pair2idxs]==i) & (xvals[pair1idxs]==j)
                
                prefs_ijk = np.sum(data[pairidxs, 3] == 2) + 0.5 * np.sum(data[pairidxs, 3] == 1) 
                prefs_ijk += np.sum(data[pair_rev_idxs, 3] == 2) + 0.5 * np.sum(data[pair_rev_idxs, 3] == 1)  
                prefs_ijk_total = float(np.sum(pairidxs) + np.sum(pair_rev_idxs))

                prefs_mean[idx] = prefs_ijk / prefs_ijk_total
                prefs_var[idx] = (1 - prefs_mean[idx])**2 * (np.sum(data[pairidxs, 3] == 2) + 
                                                             np.sum(data[pairidxs, 3] == 0) + 
                                                             np.sum(data[pair_rev_idxs, 3] == 2) + 
                                                             np.sum(data[pair_rev_idxs, 3] == 0)) + \
                                 (0.5 - prefs_mean[idx])**2 * (np.sum(data[pairidxs, 3] == 1) + 
                                                             np.sum(data[pair_rev_idxs, 3] == 1))
                prefs_var[idx] /= prefs_ijk_total
        
        x = np.arange(N**2)
        plt.plot(x, prefs_mean, label='mean for cluster %s' % k)
        
        # compute the variance
        plt.fill_between(x, prefs_mean - prefs_var, prefs_mean + prefs_var, alpha=0.4, label='variance for cluster %s' % k)    
        
    plt.legend(loc='best')
    plt.xlabel('Arguments')
    plt.ylabel('Fraction of Preferences for a > b')
    plt.savefig(plotdir + 'd7_obsprefk_%s.eps' % (label))
    
def bar_IAA_by_cluster(plotdir, ncomponents, data, cluster_labels, label, ntexts):
    U = data[:, 1] * ntexts + data[:, 2] # translate the IDs for the arguments in a pairwise comparison to a single ID
    C = data[:, 3]
    L = data[:, 0]
    IAA_all = alpha(U, C, L)    

    for c in range(len(cluster_labels)):    
        IAA = np.zeros(ncomponents + 1)
        labels = ['All People']
        for k in range(ncomponents):
            # use a set of samples from the mixture distribution to approximate the mean.
            kpersonidxs = np.argwhere(cluster_labels[c]==k)
            if not len(kpersonidxs):
                print('No people were assigned to cluster %i' % k)
            elif len(kpersonidxs) == 1:
                print('Singleton cluster %i ' % k)
            else:
                kidxs = np.in1d(data[:, 0], kpersonidxs)
                Uk = data[kidxs, 1] * ntexts + data[kidxs, 2] # translate the IDs for the arguments in a pairwise comparison to a single ID
                Ck = data[kidxs, 3] # classifications
                Lk = data[kidxs, 0] # labellers
            
                IAA[k+1] = alpha(Uk, Ck, Lk)   
            labels.append('%i' % k)
            
        IAA[0] = IAA_all
            
        _, ax = plt.subplots()
        bar_xvals = np.arange(np.sum(IAA != 0))
        ax.bar(bar_xvals, IAA[IAA != 0])
        plt.xlabel('Cluster Index')
        ax.set_xticklabels(np.array(labels)[IAA != 0])
        ax.set_xticks(np.arange(np.sum(IAA != 0)) + 0.375)
        plt.ylabel("Krippendorff's Alpha")
        plt.title('Inter-Annotator Agreement Within Each Cluster (Aff. Prop.)')
        plt.savefig(plotdir + '/IAA_%s' % (label[c]))  
        
    plt.close('all')         
    
# Factor Visualisation -----------------------------------------------------------------------------------------------
def plot_person_factor_distribution(plotdir, nflabel, nfactors, nworkers, fbar_trans):
    plt.figure()
    plt.title('Distribution of People Along Each of %i Factors' % nfactors)
    for k in range(nfactors):
        plt.subplot(nfactors / 4 + 1, 4, k+1 )
        plt.scatter(np.arange(nworkers), fbar_trans[:, k])
        plt.xlim(-5 * nworkers, 6 * nworkers)
        plt.xlabel('Indexes of People')
        plt.ylabel('Component of Preference Function Embedding')
    plt.savefig(plotdir + '/factors_individual_%s.eps' % nflabel)
    plt.close('all')
    
def plot_person_factor_pairs(plotdir, nflabel, nfactors, fbar_trans):
    # Task D4: Plot pairs of components/cluster distributions -- need some way to select pairs if we are going to do this
    plt.figure()
    plt.title('Distribution of People Along Pairs of Factors (%i Factors)' % nfactors)
    for k in range(nfactors):
        for k2 in range(nfactors):
            plt.subplot(nfactors**2 / 4 + 1, 4, k*nfactors + k2 + 1 )
            plt.scatter(fbar_trans[:, k], fbar_trans[:, k2])
            plt.xlabel('component %i' % k)
            plt.ylabel('component %i' % k2)
            plt.title('Factors %i and %i' % (k, k2))
    plt.savefig(plotdir + '/factors_pairs_%s.eps' % nflabel)    
    plt.close('all')
    
if __name__ == '__main__':

    datadir, plotdir, nx, ny, data, pair1coords, pair2coords, pair1idxs, pair2idxs, xvals, yvals, prefs, personids, \
        npairs, nworkers, ntexts = load()
        
    # Task A3  --------------------------------------------------------------------------------------------------------    
#     
#     # Task A1 continued. Put the data into the correct format for visualisation/clustering
#     

    if  'model_gponly' not in globals():
        filename = datadir + '/c1_model_gponly_%i.pkl' % (-1)
        if os.path.isfile(filename):
            with open(filename , 'rb') as fh:
                model_gponly = pickle.load(fh)
        else:
            tempresults = np.zeros((npairs, 1))
            tester = PredictionTester(datadir, -1, nx, ny, personids, pair1coords, pair2coords, prefs, 
                        np.ones(npairs), np.ones(npairs), tempresults)
            _, model_gponly = tester.run_gp_separate(-1) 

    N = model_gponly.t.shape[1]
    fbar = np.zeros(model_gponly.t.shape) # posterior means
    v = np.zeros(model_gponly.t.shape) # posterior variance
    for person in model_gponly.pref_gp:
        fbar[person, :] = model_gponly.f[person][:, 0]
        v[person, :] = model_gponly.pref_gp[person].v[:, 0]
    fstd = np.sqrt(v)
    # Section B. VISUALISING THE LATENT PREFERENCE FUNCTION AND RAW DATA WITHOUT MODELS -------------------------------
    # B1. Combine all these functions into a mixture distribution to give an overall density for the whole population
    minf = np.min(fbar - fstd) # min value to plot
    maxf = np.max(fbar - fstd) # max value to plot
    density_xvals = np.arange(minf, maxf, (maxf-minf) / 100.0 ) # 100 points to plot
    density_xvals = np.tile(density_xvals[:, np.newaxis], (1, fbar.shape[1]))
     
    findividual = np.zeros((density_xvals.shape[0], density_xvals.shape[1], nworkers))
    seenidxs = np.zeros(fbar.shape, dtype=bool)
     
    model_coords_1d = coord_arr_to_1d(model_gponly.obs_coords)
    pair1coords_1d = coord_arr_to_1d(pair1coords)
    pair2coords_1d = coord_arr_to_1d(pair2coords)
     
    for person in range(nworkers):
        pairidxs_p = personids == person
        itemidxs_p = np.in1d(model_coords_1d, pair1coords_1d[pairidxs_p]) | np.in1d(model_coords_1d, pair2coords_1d[pairidxs_p])
        #fsum[:, pidxs] += norm.cdf(density_xvals[:, pidxs], loc=fbar[person:person+1, pidxs], scale=fstd[person:person+1, pidxs])
        findividual[:, itemidxs_p, person] = norm.pdf(density_xvals[:, itemidxs_p], 
                                    loc=fbar[person:person+1, itemidxs_p], scale=fstd[person:person+1, itemidxs_p])
        seenidxs[person, itemidxs_p] = 1
     

    plot_pref_func_totals_all_people(plotdir, datadir, density_xvals, findividual)
    plot_pref_func_var_between_people(plotdir, datadir, fbar, seenidxs)
    plot_raw_prefs(plotdir, data, xvals, pair1idxs, pair2idxs, N)
    plot_pref_func_means(plotdir, fbar, N)
    
    # Section D: CLUSTER ANALYSIS -------------------------------------------------------------------------------------
    # The data to cluster is stored in fbar.
        
    # create an affinity matrix for clustering the raw data
    A = np.zeros((nworkers, nworkers))
    for i in range(ntexts):
        for j in range(ntexts):
            idxs = ((pair1idxs==i) & (pair2idxs==j)) | ((pair1idxs==j) & (pair2idxs==i))
            p_ij = prefs[idxs]
            
            k_ij0 = personids[idxs & (prefs==0)]
            k_ij05 = personids[idxs & (prefs==0.5)]
            k_ij1 = personids[idxs & (prefs==1)]
            
            # increment scores where they agree
            # decrement scores where they disagree
            for k in k_ij0:
                A[k_ij0, k] += 1
                A[k_ij05, k] -= 1
                A[k_ij1, k] -= 1
                            
            for k in k_ij05:
                A[k_ij05, k] += 1
                A[k_ij0, k] -= 1
                A[k_ij1, k] -= 1
                        
            for k in k_ij1:    
                A[k_ij1, k] += 1
                A[k_ij0, k] -= 1
                A[k_ij05, k] -= 1                                        
    
    # Compute the raw data as a matrix of workers x pairs of items
    fraw_pos = coo_matrix(( 2*(prefs-0.5), (personids, pair1idxs*ntexts + pair2idxs)), shape=(nworkers, ntexts*ntexts), dtype=int)
    fraw_neg = coo_matrix(( 2*(0.5-prefs), (personids, pair2idxs*ntexts + pair1idxs)), shape=(nworkers, ntexts*ntexts), dtype=int)
    fraw = fraw_pos + fraw_neg        
    fraw = fraw[:, np.squeeze(np.asarray((fraw!=0).sum(axis=0)>0))].toarray()       
    
                         
    # Task D1 -----------------------------------------------------------------------------------------------------
#     ncomponents = 20
#     #gmm = GaussianMixture(n_components=ncomponents)
#     gmm = BayesianGaussianMixture(n_components=ncomponents, weight_concentration_prior=1.0/10) #DPGMM(nfactors)
#     gmm.fit(fbar)
#     gmm_labels = gmm.predict(fbar)
#     gmm_proba = gmm.predict_proba(fbar) # why is this returning only binary values?  
#      
    gmm_raw = BayesianGaussianMixture(n_components=ncomponents, weight_concentration_prior=(1.0 / 20) * 10) #DPGMM(nfactors)
    gmm_raw.fit(fraw)
    gmm_raw_labels = gmm_raw.predict(fraw)
    gmm_raw_proba = gmm_raw.predict_proba(fraw) 
# 
#     gmm_membership = np.sum(gmm_proba, axis=0)
#     gmm_membership_greedy = np.zeros(ncomponents)
# 
#     for k in range(ncomponents):
#         gmm_membership_greedy[k] = np.sum(gmm_labels==k)
#     label=['GMM (greedy)', 'GMM (probabilities)']                      
                    
    # Task D1b ----------------------------------------------------------------------------------------------------
    # Try additional clustering algorithms from SKLearn. 
    # Spectral clustering may be inappropriate because it clusters connected data rather than ensuring that all 
    # members of a cluster are very similar. 
    # Affinity propagation finds exemplars -- perhaps more appropriate here? May allow for more fuzzy clusters.
    # A lot of outliers in their own singleton clusters -- should we cut them out and ignore?
    afprop = AffinityPropagation()
    afprop_labels = afprop.fit_predict(fbar)
    ncomponents = len(np.unique(afprop_labels))

    # cannot apply this to factor analysis -- can do another plot to show distributions. How fuzzy are memberships?        
    afprop_membership = np.zeros(ncomponents)
    membership_weights_ap = np.zeros((ncomponents, N))
    for k in range(ncomponents):
        afprop_membership[k] = np.sum(afprop_labels==k)
        membership_weights_ap[k] = (afprop_labels==k)

    # doesn't really work -- almost all end up in one cluster + one other small cluster + a few outliers
    #agg = AgglomerativeClustering(n_clusters=10)
    #agg_labels = agg.fit_predict(fbar)

    # Task D1c ----------------------------------------------------------------------------------------------------
    # Clustering the original data rather than the expected function mean
    ap_rawdata = AffinityPropagation(affinity='precomputed')
    apraw_labels = ap_rawdata.fit_predict(A)
    ncomponents = len(np.unique(apraw_labels))
    
    apraw_membership = np.zeros(ncomponents)
    membership_weights_apraw = np.zeros((ncomponents, N))
    for k in range(ncomponents):
        apraw_membership[k] = np.sum(apraw_labels==k)
        membership_weights_apraw[k] = (apraw_labels==k)
        
    # doesn't really work -- almost all end up in one cluster + a few outliers
    # aggraw = AgglomerativeClustering(n_clusters=10)
    # aggraw_labels = aggraw.fit_predict(fraw)   
     
    # Visualise D1 ------------------------------------------------------------------------------------------------

#         plt.bar(np.arange(ncomponents), gmm_membership_greedy, label='GMM (greedy)')     
#         plt.bar(np.arange(ncomponents), gmm_membership, label='GMM (probabilities)')
    bar_cluster_membership(plotdir, len(np.unique(afprop_labels)), [afprop_membership], 
                                           ['Aff. Prop.'])
    
    # Task D5, D6: For each cluster, plot f-value means and variance within the clusters --------------------------
    plot_pref_func_stats_by_cluster(plotdir, len(np.unique(afprop_labels)), 
                                                    membership_weights_ap, 'Aff. Prop.', density_xvals, findividual)          
    # Task D7: For each cluster, plot variance in observed prefs in each cluster ----------------------------------  
    plot_raw_pref_stats_by_cluster(plotdir, len(np.unique(afprop_labels)), membership_weights_ap, 
                                                   'Aff. Prop.', data, xvals, pair1idxs, pair2idxs, N)   
    # Task D8: For each cluster, plot variance in predicted prefs in each cluster ---------------------------------
    plot_predicted_pref_stats_by_cluster(plotdir, ncomponents, membership_weights_ap, 
                                                    'Aff. Prop.', fbar, N)
    
    plot_pref_func_stats_by_cluster(plotdir, len(np.unique(apraw_labels)), 
                                membership_weights_apraw, 'Aff. Prop. on Raw Data', density_xvals, findividual) 
    plot_raw_pref_stats_by_cluster(plotdir, len(np.unique(apraw_labels)), 
                                membership_weights_apraw, 'Aff. Prop. on Raw Data',  data, xvals, pair1idxs, pair2idxs, N)
    plot_predicted_pref_stats_by_cluster(plotdir, ncomponents, membership_weights_apraw, 
                                                    'Aff. Prop. on Raw Data', fbar, N)
        
    bar_IAA_by_cluster(plotdir, ncomponents, data, [afprop_labels, apraw_labels], 
                                       ['afprop', 'apraw'], ntexts)

    nfactors_list = [3, 5, 10]
    for nfactors in nfactors_list:
        
        nflabel = 'nfactors_%i' % nfactors # an extra label to add to plots and filenames
        
        # Task D2 -----------------------------------------------------------------------------------------------------
        fa = FactorAnalysis(nfactors)
        fbar_trans = fa.fit_transform(fbar)
        
        faraw = FactorAnalysis(nfactors)
        faraw_trans = faraw.fit_transform(fraw)
        
        # Task D3 -----------------------------------------------------------------------------------------------------       
        
        plot_person_factor_distribution(plotdir, nflabel, nfactors, nworkers, fbar_trans)
        plot_person_factor_pairs(plotdir, nflabel, nfactors, fbar_trans)