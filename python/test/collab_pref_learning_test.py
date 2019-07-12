'''
Test the preference_features module with some simple synthetic data test

Created on 3 Mar 2017

@author: edwin
'''
import logging
import os
import sys

logging.basicConfig(level=logging.DEBUG)

sys.path.append("./python")
sys.path.append("./python/analysis")
sys.path.append("./python/models")
sys.path.append("./python/analysis/lukin_comparison")

import numpy as np
from gp_classifier_vb import matern_3_2_from_raw_vals, coord_arr_to_1d
from scipy.stats import multivariate_normal as mvn
from scipy.linalg import block_diag
from gp_pref_learning_test import gen_synthetic_prefs
from collab_pref_learning_vb import CollabPrefLearningVB
from collab_pref_learning_svi import CollabPrefLearningSVI


def gen_synthetic_personal_prefs(Nfactors, nx, ny, N, Npeople, P, ls, s, lsy, Npeoplefeatures=4):

    pair1idxs = []
    pair2idxs = []
    prefs = []
    personids = []

    # generate a common prior:
    xvals = np.tile(np.arange(nx)[:, np.newaxis], (1, ny)).flatten().astype(float)
    yvals = np.tile(np.arange(ny)[np.newaxis, :], (nx, 1)).flatten().astype(float)
    Kt = matern_3_2_from_raw_vals(np.concatenate((xvals[:, np.newaxis], yvals[:, np.newaxis]), axis=1), ls) / s
    # t = np.zeros((nx * ny, 1))#
    t = mvn.rvs(cov=Kt).reshape(nx * ny, 1)

    Kw = [Kt for _ in range(Nfactors)]
    Kw = block_diag(*Kw)
    w = mvn.rvs(cov=Kw).reshape(Nfactors, nx * ny).T

    # person_features = None
    person_features = np.zeros((Npeoplefeatures, Npeople))
    for i in range(Npeoplefeatures):
        person_features[i, :int(Npeople / 2)] = -0.2
        person_features[i, int(Npeople / 2):] = 0.2
        person_features[i, :] += np.arange(Npeople)

    Ky = matern_3_2_from_raw_vals(person_features.T, lsy) / s
    Ky = [Ky for _ in range(Nfactors)]
    Ky = block_diag(*Ky)
    y = mvn.rvs(cov=Ky).reshape(Nfactors, Npeople)

    f_all = w.dot(y) + t

    for p in range(Npeople):
        f_p = f_all[:, p].reshape(nx, ny)

        if p == 0:
            _, prefs_p, item_features, pair1idxs_p, pair2idxs_p, _, K = gen_synthetic_prefs(f_pre=f_p, nx=nx, ny=ny, N=N,
                                                                                            P=P, s=s, ls=ls)
        else:
            _, prefs_p, _, pair1idxs_p, pair2idxs_p, _, K = gen_synthetic_prefs(f_pre=f_p, nx=nx, ny=ny, N=N, P=P,
                                                                                s=s, ls=ls, item_features=item_features)

        pair1idxs = np.concatenate((pair1idxs, pair1idxs_p)).astype(int)
        pair2idxs = np.concatenate((pair2idxs, pair2idxs_p)).astype(int)
        prefs = np.concatenate((prefs, prefs_p)).astype(int)
        personids = np.concatenate((personids, np.zeros(len(pair1idxs_p)) + p)).astype(int)

    _, uidxs, inverseidxs = np.unique(coord_arr_to_1d(item_features), return_index=True, return_inverse=True)
    item_features = item_features[uidxs]
    pair1idxs = inverseidxs[pair1idxs]
    pair2idxs = inverseidxs[pair2idxs]

    # return t as a grid
    t = t.reshape(nx, ny)

    return prefs, item_features, person_features, pair1idxs, pair2idxs, personids, f_all, w, t, y

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.DEBUG)    

    fix_seeds = True
    do_profiling = False
    
    if do_profiling:
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()

    # make sure the simulation is repeatable
    if fix_seeds:
        np.random.seed(10)

    logging.info( "Testing Bayesian preference components analysis using synthetic data..." )
    
    if 'item_features' not in globals():
        #         Npeople = 20
        #         N = 25
        #         P = 100 # pairs per person in test+training set
        #         nx = 5
        #         ny = 5

        Npeople = 8
        N = 16
        P = 5000
        nx = 4
        ny = 4

        Npeoplefeatures = 3
        ls = [10, 5]
        s = 0.0001
        lsy = 2 + np.zeros(Npeoplefeatures)
        Nfactors = 2

        prefs, item_features, person_features, pair1idxs, pair2idxs, personids, latent_f, w, t, y = \
            gen_synthetic_personal_prefs(Nfactors, nx, ny, N, Npeople, P, ls, s, lsy, Npeoplefeatures)

        Ptest_percent = 0.2
        Ptest = int(Ptest_percent * pair1idxs.size)
        testpairs = np.random.choice(pair1idxs.shape[0], Ptest, replace=False)
        testidxs = np.zeros(pair1idxs.shape[0], dtype=bool)
        testidxs[testpairs] = True
        trainidxs = np.invert(testidxs)
    
    # if fix_seeds:
    #     np.random.seed() # do this if we want to use a different seed each time to test the variation in results
        
    # Model initialisation --------------------------------------------------------------------------------------------
    if len(sys.argv) > 1:
        use_svi = sys.argv[1] == 'svi'
    else:
        use_svi = True
    use_t = True
    use_person_features = True
    optimize = False

    ls_initial = np.array(ls)# + np.random.rand(len(ls)) * 10)
    print(("Initial guess of length scale for items: %s, true length scale is %s" % (ls_initial, ls)))
    lsy_initial = np.array(lsy)# + np.random.rand(len(lsy)) * 10)# + 7
    print(("Initial guess of length scale for people: %s, true length scale is %s" % (lsy_initial, lsy)))
    if use_svi:
        model = CollabPrefLearningSVI(2, Npeoplefeatures if use_person_features else 0, ls=ls_initial,
                                      lsy=lsy_initial, use_common_mean_t=use_t,
                                      nfactors=7, forgetting_rate=0.7, ninducing=16, max_update_size=100, use_lb=True)
    else:
        model = CollabPrefLearningVB(2, Npeoplefeatures if use_person_features else 0, ls=ls_initial, lsy=lsy_initial,
                                     use_common_mean_t=use_t, nfactors=7, use_lb=True)

    if fix_seeds:
        np.random.seed(22)

    model.verbose = False
    model.min_iter = 1
    model.max_iter = 200
    model.fit(personids[trainidxs], pair1idxs[trainidxs], pair2idxs[trainidxs], item_features, prefs[trainidxs], 
              person_features.T if use_person_features else None, optimize=optimize)
#               None, optimize=True)    
    print(("Difference between true item length scale and inferred item length scale = %s" % (ls - model.ls)))
    print(("Difference between true person length scale and inferred person length scale = %s" % (lsy - model.lsy)))
    
    # turn the values into predictions of preference pairs.
    results = model.predict(personids[testidxs], pair1idxs[testidxs], pair2idxs[testidxs], item_features,
                            person_features.T if use_person_features else None)
    
    # make the test more difficult: we predict for a person we haven't seen before who has same features as another
    result_new_person = model.predict(
        [np.max(personids) + 1], pair1idxs[testidxs][0:1], pair2idxs[testidxs][0:1],
        item_features,
        np.concatenate((person_features.T, person_features[:, personids[0:1]].T), axis=0) if use_person_features
        else None)
    print("Test using new person: %.3f" % result_new_person)
    print("Old prediction: %.3f" % results[0])

    print("Testing prediction of new + old people")
    result_new_old_person = model.predict(
        np.concatenate((personids[testidxs], [np.max(personids) + 1])),
        np.concatenate((pair1idxs[testidxs], pair1idxs[testidxs][0:1])),
        np.concatenate((pair2idxs[testidxs], pair2idxs[testidxs][0:1])),
        item_features,
        np.concatenate((person_features.T, person_features[:, personids[0:1]].T), axis=0) if use_person_features
        else None)
    print("Test using new person while predicting old people: %.3f" % result_new_old_person[-1])
    #print("Result is correct = " + str(np.abs(results[0] - result_new_person) < 1e-6) 
    
    if do_profiling:
        pr.disable()
        import datetime
        pr.dump_stats('preference_features_test_svi_%i_%s.profile' % (use_svi, datetime.datetime.now()))
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print((s.getvalue()))
    
    # To make sure the simulation is repeatable, re-seed the RNG after all the stochastic inference has been completed
    if fix_seeds:
        np.random.seed(2)    
    
    from sklearn.metrics import accuracy_score
    
    p_pred = results
    p_pred_round = np.round(results).astype(int)
    p = prefs[testidxs]
       
    print(" --- Preference prediction metrics --- " )
    print(("Brier score of %.3f" % np.sqrt(np.mean((p-p_pred)**2))))
    p_pred[p_pred > (1-1e-6)] = 1 - 1e-6
    p_pred[p_pred < 1e-6] = 1e-6
    print(("Cross entropy error of %.3f" % -np.mean(p * np.log(p_pred) + (1-p) * np.log(1 - p_pred))))
            
    from sklearn.metrics import f1_score, roc_auc_score
    print(("F1 score of %.3f" % f1_score(p, p_pred_round)))
    print(('Accuracy: %f' % accuracy_score(p, p_pred_round)))
    print(("ROC of %.3f" % roc_auc_score(p, p_pred)))

    print(" --- Latent item feature prediction metrics --- " )
    
    # get the w values that correspond to the coords seen by the model
    widxs = np.ravel_multi_index((
           model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)), dims=(nx, ny))
    w = w[widxs, :]
    
    # how can we handle the permutations of the features?
    #scipy.factorial(model.Nfactors) / scipy.factorial(model.Nfactors - w.shape[1])
    # remove the features from the model with least variation -- these are the dead features
    wvar = np.var(model.w, axis=0)
    chosen_features = np.argsort(wvar)[-w.shape[1]:]
    w_pred = model.w[:, chosen_features].T.reshape(w.shape[1] * N)

    # w_pred_cov = model.w_cov.reshape(model.Nfactors * N, model.Nfactors, N)
    # w_pred_cov = np.swapaxes(w_pred_cov, 0, 2).reshape(N, model.Nfactors, model.Nfactors, N)
    # w_pred_cov = w_pred_cov[:, chosen_features, :, :][:, :, chosen_features, :]
    # w_pred_cov = w_pred_cov.reshape(N, w.shape[1], w.shape[1] * N)
    # w_pred_cov = np.swapaxes(w_pred_cov, 0, 2).reshape(w.shape[1] * N, w.shape[1] * N)
    #
    # print("w: RMSE of %.3f" % np.sqrt(np.mean((w.T.reshape(N * w.shape[1])-w_pred)**2)))
    # print("w: NLPD error of %.3f" % -mvn.logpdf(w.T.reshape(N * w.shape[1]), mean=w_pred, cov=w_pred_cov))
    #
    # print(" --- Latent person feature prediction metrics --- ")
    #
    # yvar = np.var(model.y, axis=1)
    # chosen_features = np.argsort(yvar)[-y.shape[0]:]
    # y_pred = model.y[chosen_features, :].reshape(y.shape[0] * Npeople)
    #
    # y_pred_cov = model.y_cov.reshape(model.Nfactors * Npeople, model.Nfactors, Npeople)
    # y_pred_cov = np.swapaxes(y_pred_cov, 0, 2).reshape(Npeople, model.Nfactors, model.Nfactors, Npeople)
    # y_pred_cov = y_pred_cov[:, chosen_features, :, :][:, :, chosen_features, :]
    # y_pred_cov = y_pred_cov.reshape(Npeople, w.shape[1], w.shape[1] * Npeople)
    # y_pred_cov = np.swapaxes(y_pred_cov, 0, 2).reshape(w.shape[1] * Npeople, w.shape[1] * Npeople)
    #
    # print("y: RMSE of %.3f" % np.sqrt(np.mean((y.reshape(Npeople * w.shape[1])-y_pred)**2)))
    # print("y: NLPD error of %.3f" % -mvn.logpdf(y.reshape(Npeople * w.shape[1]), mean=y_pred, cov=y_pred_cov))
            
#     from scipy.stats import kendalltau
#      
#     for p in range(Npeople):
#         logging.debug( "Personality features of %i: %s" % (p, str(model.w[p])) )
#         for q in range(Npeople):
#             logging.debug( "Distance between personalities: %f" % np.sqrt(np.sum(model.w[p] - model.w[q])**2)**0.5 )
#             logging.debug( "Rank correlation between preferences: %f" %  kendalltau(model.f[p], model.f[q])[0] )
    
    # visualise the results
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap('jet')                
    cmap._init()    
    
    # t
    fig = plt.figure()
    tmap = np.zeros((nx, ny))
    tmap[model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)] = model.t.flatten()
    scale = np.std(tmap)
    if scale == 0:
        scale = 1
    tmap /= scale
    ax = plt.imshow(tmap, cmap=cmap, aspect=None, origin='lower', \
                   vmin=np.min(tmap), vmax=np.max(tmap), interpolation='none', filterrad=0.01)
    plt.title('predictions at training points: t (item mean)')
    fig.colorbar(ax)

#     plt.figure()
#     tmap = np.zeros((nx, ny))
#     tmap[model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)] = np.sqrt(np.diag(model.t_cov))
#     scale = np.std(tmap[model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)])
#     plt.imshow(tmap, cmap=cmap, aspect=None, origin='lower', \
#                    vmin=-scale*2, vmax=scale*2, interpolation='none', filterrad=0.01)
#     plt.title('STD at training points: t (item mean)')

    fig = plt.figure()
    tmap = np.zeros((nx, ny))
    tmap[model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)] = t[model.obs_coords[:, 0].astype(int),
                                                                      model.obs_coords[:, 1].astype(int)].flatten()
    scale = np.std(tmap)
    if scale == 0:
        scale = 1
    tmap /= scale
    ax = plt.imshow(tmap, cmap=cmap, aspect=None, origin='lower', vmin=np.min(tmap), vmax=np.max(tmap), interpolation='none', filterrad=0.01)
    plt.title('ground truth at training points: t (item mean)')
    fig.colorbar(ax)

    # y
    fig = plt.figure()
    ymap = model.y.T
    scale = np.std(ymap)
    if scale == 0:
        scale = 1.0
    #scale = np.sqrt(model.rate_sy[np.newaxis, :]/model.shape_sy[np.newaxis, :])
    ymap /= scale
    ax = plt.imshow(ymap, cmap=cmap, origin='lower', extent=[0, ymap.shape[1], 0, ymap.shape[0]],
               aspect=Nfactors / float(ymap.shape[0]), vmin=np.min(ymap), vmax=np.max(ymap), interpolation='none', filterrad=0.01)
    plt.title('predictions at training points: y (latent features for people)')
    fig.colorbar(ax)

    fig = plt.figure()
    ymap = y.T
    scale = np.std(ymap)
    if scale == 0:
        scale = 1.0
    ymap /= scale
    ax = plt.imshow(ymap, cmap=cmap, origin='lower', extent=[0, ymap.shape[1], 0, ymap.shape[0]],
               aspect=Nfactors / float(ymap.shape[0]), vmin=np.min(ymap), vmax=np.max(ymap), interpolation='none', filterrad=0.01)
    plt.title('ground truth at training points: y (latent features for people')
    fig.colorbar(ax)
       
    # w
    scale = np.std(model.w)
    if scale == 0:
        scale = 1.0
    model.w /= scale
    wmin = np.min(model.w)
    wmax = np.max(model.w)

    for f in range(model.Nfactors):
        fig = plt.figure()
        wmap = np.zeros((nx, ny))
        wmap[model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)] = model.w[:, f]
        ax = plt.imshow(wmap, cmap=cmap, origin='lower', extent=[0, wmap.shape[1], 0, wmap.shape[0]],
                   aspect=None, vmin=wmin, vmax=wmax, interpolation='none', filterrad=0.01)
        plt.title('predictions at training points: w_%i (latent feature for items)' %f)
        fig.colorbar(ax)

#         fig = plt.figure()
#         wmap = np.zeros((nx, ny))
#         wmap[model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)] = np.sqrt(model.w_cov[np.arange(model.N*f, model.N*(f+1)), 
#                                                                                    np.arange(model.N*f, model.N*(f+1))])        
#         scale = np.std(wmap[model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)])
#         wmap /= scale
#         ax = plt.imshow(wmap, cmap=cmap, origin='lower', extent=[0, wmap.shape[1], 0, wmap.shape[0]], aspect=None, vmin=-2,
#                    vmax=2, interpolation='none', filterrad=0.01)
#         plt.title('STD at training points: w_%i (latent feature for items)' %f)
#         fig.colorbar(ax)
    scale = np.std(w)
    if scale == 0:
        scale = 1.0
    w /= scale
    wmin = np.min(w)
    wmax = np.max(w)

    for f in range(Nfactors):
        fig = plt.figure()
        wmap = np.zeros((nx, ny))
        wmap[model.obs_coords[:, 0].astype(int), model.obs_coords[:, 1].astype(int)] = w[:, f]

        ax = plt.imshow(wmap, cmap=cmap, origin='lower', extent=[0, wmap.shape[1], 0, wmap.shape[0]],
                   aspect=None, vmin=wmin, vmax=wmax, interpolation='none', filterrad=0.01)
        plt.title('ground truth at training points: w_%i (latent feature for items' % f)
        fig.colorbar(ax)

    plt.show()