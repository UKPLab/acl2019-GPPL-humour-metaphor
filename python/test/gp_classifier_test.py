'''
Created on 3 Mar 2017

@author: edwin
'''
import logging
logging.basicConfig(level=logging.DEBUG)
import numpy as np
from gp_classifier_vb import matern_3_2_from_raw_vals, coord_arr_to_1d, sigmoid, GPClassifierVB
from gp_classifier_svi import GPClassifierSVI
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import kendalltau

def gen_synthetic_classifications(f_prior_mean=None, nx=100, ny=100):
    # f_prior_mean should contain the means for all the grid squares
    
    # Generate some data
    ls = [10, 10]
    sigma = 0.1 
    N = 2000
    #C = 100 # number of labels for training
    s = 1 # inverse precision scale for the latent function.
    
    # Some random feature values
    xvals = np.random.choice(nx, N, replace=True)[:, np.newaxis]
    yvals = np.random.choice(ny, N, replace=True)[:, np.newaxis]
    # remove repeated coordinates
    for coord in range(N):
        while np.sum((xvals==xvals[coord]) & (yvals==yvals[coord])) > 1:
            xvals[coord] = np.random.choice(nx, 1)
            yvals[coord] = np.random.choice(ny, 1)           
        
    K = matern_3_2_from_raw_vals(np.concatenate((xvals, yvals), axis=1), ls)
    if f_prior_mean is None:
        f = mvn.rvs(cov=K/s) # zero mean        
    else:
        f = mvn.rvs(mean=f_prior_mean[xvals, yvals].flatten(), cov=K/s) # zero mean
    
    # generate the noisy function values for the pairs
    fnoisy = norm.rvs(scale=sigma, size=N) + f
    
    # generate the discrete labels from the noisy function
    labels = np.round(sigmoid(fnoisy))
    
    return N, nx, ny, labels, xvals, yvals, f, K

if __name__ == '__main__':
    fix_seeds = True
    
    # make sure the simulation is repeatable
    if fix_seeds:
        np.random.seed(1)
    
    N, nx, ny, labels, xvals, yvals, f, K = gen_synthetic_classifications()
        
    # separate training and test data
    Ctest = int(len(labels) * 0.1)
    testids = np.random.choice(labels.shape[0], Ctest, replace=False)
    testidxs = np.zeros(labels.shape[0], dtype=bool)
    testidxs[testids] = True
    trainidxs = np.invert(testidxs)
    
    xvals_test = xvals[testidxs].flatten()
    yvals_test = yvals[testidxs].flatten()
    _, uidxs = np.unique(coord_arr_to_1d(np.concatenate((xvals_test[:, np.newaxis], yvals_test[:, np.newaxis]), axis=1)), 
              return_index=True)
    xvals_test = xvals_test[uidxs][:, np.newaxis]
    yvals_test = yvals_test[uidxs][:, np.newaxis]
    f_test = f[testidxs][uidxs]
    
    models = {}
    
    ls_initial = [112]#np.random.randint(1, 100, 2)#[10, 10] 
    
    model = GPClassifierVB(2, z0=0.5, shape_s0=1, rate_s0=1, ls_initial=ls_initial)
    model.verbose = True
    model.max_iter_VB = 1000
    model.min_iter_VB = 5
    model.uselowerbound = True
    model.delay = 1
    #model.conv_threshold_G = 1e-8
    #model.conv_check_freq = 1
    #model.conv_threshold = 1e-3 # the difference must be less than 1% of the value of the lower bound

    models['VB'] = model

    model = GPClassifierSVI(2, z0=0.5, shape_s0=1, rate_s0=1, ls_initial=ls_initial, use_svi=True)
    model.verbose = True
    model.max_iter_VB = 1000
    model.min_iter_VB = 5
    model.uselowerbound = True
    model.delay = 1
    #model.conv_threshold_G = 1e-8
    #model.conv_check_freq = 1
    #model.conv_threshold = 1e-3 # the difference must be less than 1% of the value of the lower bound
       
    #models['SVI'] = model
    
    model = GPClassifierSVI(2, z0=0.5, shape_s0=1, rate_s0=1, ls_initial=ls_initial, use_svi=False)
    model.verbose = True
    model.max_iter_VB = 1000
    model.min_iter_VB = 5
    model.uselowerbound = True
    model.delay = 1
    #model.conv_threshold_G = 1e-8
    #model.conv_check_freq = 1
    #model.conv_threshold = 1e-3 # the difference must be less than 1% of the value of the lower bound
       
    #models['SVI_switched_off'] = model
    
    # if fix_seeds:
    #     np.random.seed() # do this to test the variation in results due to stochastic methods with same data
    
    obs_coords = np.concatenate((xvals, yvals), axis=1)
    
    for modelkey in models:
        print("--- Running model %s ---" % modelkey)
        
        model = models[modelkey]

        model.fit(obs_coords[trainidxs, :], labels[trainidxs], optimize=False)
        print("Final lower bound: %f" % model.lowerbound())
        
        # Predict at the test locations
        fpred, vpred = model.predict_f(np.concatenate((xvals_test, yvals_test), axis=1))
        
        # Compare the observation point values with the ground truth
        obs_coords_1d = coord_arr_to_1d(model.obs_coords)
        test_coords_1d = coord_arr_to_1d(np.concatenate((xvals, yvals), axis=1))
        f_obs = [f[(test_coords_1d==obs_coords_1d[i]).flatten()][0] for i in range(model.obs_coords.shape[0])]
        print("Kendall's tau (observations): %.3f" % kendalltau(f_obs, model.obs_f.flatten())[0])
            
        # Evaluate the accuracy of the predictions
        #print "RMSE of %f" % np.sqrt(np.mean((f-fpred)**2))
        #print "NLPD of %f" % -np.sum(norm.logpdf(f, loc=fpred, scale=vpred**0.5))
        print("Kendall's tau (test): %.3f" % kendalltau(f_test, fpred)[0] )
            
        rho = sigmoid(f[testidxs])
        rho_pred, var_rho_pred = model.predict(obs_coords[testidxs], variance_method='sample')
        rho_pred = rho_pred.flatten()
        t_pred = np.round(rho_pred)
        
        # To make sure the simulation is repeatable, re-seed the RNG after all the stochastic inference has been completed
        if fix_seeds:
            np.random.seed(2)    
        
        print("Brier score of %.3f" % np.sqrt(np.mean((rho-rho_pred)**2)) )
        print("Cross entropy error of %.3f" % -np.sum(rho * np.log(rho_pred) + (1-rho) * np.log(1 - rho_pred)) )
        
        t = np.round(rho)
        
        from sklearn.metrics import f1_score, roc_auc_score
        print("F1 score of %.3f" % f1_score(t, t_pred) )
        print("Accuracy of %.3f" % np.mean(t==t_pred) )
        print("ROC of %.3f" % roc_auc_score(t, rho_pred) )
