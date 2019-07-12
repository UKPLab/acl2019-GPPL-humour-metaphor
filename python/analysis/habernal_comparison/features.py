'''
Created on 1 Jun 2017

Load a set of feature lengthscales from a good run with 'both' types of features. 
Sort them by lengthscale.
Plot the distribution.

Identify which type of feature they are: add colours or markers to the plot.

Provide a zoomed-in variant for the best 25 features.

@author: simpson
'''

import os, pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tests import load_train_test_data, load_embeddings, load_ling_features, get_fold_data, TestRunner
from matplotlib.ticker import MaxNLocator

if __name__ == '__main__':

    expt_folder_name = 'crowdsourcing_argumentation_opt/'
    #expt_folder_name = 'crowdsourcing_argumentation_expts/'

    dataset = 'UKPConvArgStrict'#'UKPConvArgAll_evalMACE'#
    methods = ['SinglePrefGP_weaksprior_1104']#'SinglePrefGP_weaksprior_2107', 'SinglePrefGP_weaksprior_0308', 'SinglePrefGP_weaksprior_1004desktop169']
    feature_type = 'both'
    embeddings_type = 'word_mean'
    di = 0.00

    selected_folds_all = [[0, 1, 6, 12, 13]]#failed folds for 1104: [9, 10, 16]]
    # selected_folds_all = [[0, 2, 6, 7, 10, 14, 16, 17, 22, 24, 28, 30],  # for the 2107 dataset
    #                       [5, 9, 12, 13, 21, 25, 27],
    #                       [1, 4, 8, 21, 29]]

    original_fold_order_file = './results/feature_analysis/foldorder_old.txt'
    o_fold_order = np.genfromtxt(os.path.expanduser(original_fold_order_file), dtype=str)

    mean_ls = None

    for m, method in enumerate(methods):

        data_root_dir = os.path.expanduser("~/data/personalised_argumentation/")
        resultsfile_template = 'habernal_%s_%s_%s_%s_acc%.2f_di%.2f'

        resultsfile = data_root_dir + 'outputdata/' + expt_folder_name + \
            resultsfile_template % (dataset, method,
            feature_type, embeddings_type, 1.0, di) + '_test.pkl'

        resultsdir = data_root_dir + 'outputdata/' + expt_folder_name + \
            resultsfile_template % (dataset, method,
            feature_type, embeddings_type, 1.0, di)

        foldorderfile = None
        if foldorderfile is not None:
            fold_order = np.genfromtxt(os.path.expanduser(foldorderfile),
                                                        dtype=str)
        elif os.path.isfile(resultsdir + '/foldorder.txt'):
            fold_order = np.genfromtxt(os.path.expanduser(resultsdir + '/foldorder.txt'),
                                                        dtype=str)
        else:
            fold_order = None

        selected_folds = selected_folds_all[m]
        nFolds = len(selected_folds)

        if os.path.isfile(resultsfile):

            with open(resultsfile, 'r') as fh:
                data = pickle.load(fh)

            if nFolds < 1:
                nFolds = len(data[0])
        else:
            data = None

        min_folds = 0

        # Sort the features by their ID.
        # If we have discarded some features that were all zeros, the current index will not be the original feature idx.
        # How to map them back? Reload the original data and find out which features were discarded.

        folds, folds_regression, word_index_to_embeddings_map, word_to_indices_map, index_to_word_map = load_train_test_data(dataset)
        word_embeddings = load_embeddings(word_index_to_embeddings_map)
        ling_feat_spmatrix, docids = load_ling_features(dataset)

        #default_ls_value = compute_lengthscale_heuristic(feature_type, embeddings_type, word_embeddings,
        #                                         ling_feat_spmatrix, docids, folds, index_to_word_map)


        for o_foldidx, o_fold in enumerate(o_fold_order):

            if o_foldidx not in selected_folds:
                continue

            if fold_order is None: # fall back to the order on the current machine
                foldidx = np.argwhere(np.array(list(folds.keys())) == o_fold)[0][0]
                fold = list(folds.keys())[foldidx]
            else:
                foldidx = np.argwhere(fold_order == o_fold)[0][0]
                fold = fold_order[foldidx]
                if fold[-2] == "'" and fold[0] == "'":
                    fold = fold[1:-2]
                elif fold[-1] == "'" and fold[0] == "'":
                    fold = fold[1:-1]
                fold_order[foldidx] = fold

            # look for new-style data in separate files for each fold. Prefer new-style if both are found.
            foldfile = resultsdir + '/fold%i.pkl' % foldidx
            if os.path.isfile(foldfile):
                with open(foldfile, 'rb') as fh:
                    data_f = pickle.load(fh, encoding='latin1')
            else: # convert the old stuff to new stuff
                if data is None:
                    min_folds = foldidx+1
                    print('Skipping fold with no data %i' % foldidx)
                    print("Skipping results for %s, %s, %s, %s" % (method,
                                                                   dataset,
                                                                   feature_type,
                                                                   embeddings_type))
                    print("Skipped filename was: %s, old-style results file would be %s" % (foldfile,
                                                                                            resultsfile))
                    continue

                if not os.path.isdir(resultsdir):
                    os.mkdir(resultsdir)
                data_f = []
                for thing in data:
                    if foldidx in thing:
                        data_f.append(thing[foldidx])
                    else:
                        data_f.append(thing)
                with open(foldfile, 'wb') as fh:
                    pickle.dump(data_f, fh)

            trainids_a1, trainids_a2, prefs_train, personIDs_train, testids_a1, testids_a2, prefs_test, personIDs_test, \
                                                                        X, uids, utexts = get_fold_data(folds, fold, docids)

            # get the embedding values for the test data -- need to find embeddings of the whole piece of text
            runner = TestRunner('crowdsourcing_argumentation_expts_first_submission', [dataset], [feature_type],
                                [embeddings_type], [method], 0)
            runner.embeddings = word_embeddings
            runner.X = X
            runner.ling_feat_spmatrix = ling_feat_spmatrix
            runner.load_features(feature_type, embeddings_type, trainids_a1, trainids_a2, uids)
            items_feat = runner.items_feat
            valid_feats = runner.valid_feats

            min_vals = np.min(items_feat, axis=0)
            max_vals = np.max(items_feat, axis=0)

            nfeats = len(valid_feats)
            # take the mean ls for each feature across the folds
            if mean_ls is None:
                mean_ls = np.zeros(nfeats, dtype=float)
                totals = np.zeros(nfeats, dtype=int)

            #print "Warning: not computing means."
            learned_ls = data_f[7]
            initial_ls = data_f[5] #/ float(len(valid_feats)) # we want the data relative to the median -- the initial LS were also scaled by no. features
            mean_ls[valid_feats] += learned_ls / initial_ls # normalisation in original drafts
            norm_ls = learned_ls / (max_vals - min_vals)
            #mean_ls[valid_feats] += norm_ls

            print("Max normed l: %f" % np.max(norm_ls))
            totals[valid_feats] += 1
         
    #mean_ls = mean_ls[valid_feats]
    #totals = totals[valid_feats]
    mean_ls[totals != 0] = mean_ls[totals != 0] / totals[totals != 0]
    
    if feature_type == 'debug':
        feat_cats = np.array(['one', 'two', 'three'])
        featnames = feat_cats
        col = np.array(['r', 'lightgreen', 'b'])
        marks = np.array(['2', 'p', '^'])
        nembeddings = 3
    else:
        # assign category labels to each feature
        feat_cats = np.empty(nfeats, dtype=object)
        nembeddings = word_embeddings.shape[1]
        feat_cats[:nembeddings] = "embeddings"
        
        catnames = np.array(['embeddings', '_pos_ngram', 'ProductionRule', 'Rate', 'CONTEXTUALITY_MEASURE_FN',
             'ExclamationRatio', 'upperCaseRatio', 'Ratio', 'DependencyTreeDepth', 'Modal',
             'sentiment', 'oovWordsCount', 'spell_skill', '_length', 'word_more', 'Ending', 'ner.type.', '_'])
        special_catnames = np.array(['flesch', 'coleman', 'ari'])

        marks = np.array(['2', 'p', '^', 'H', 'x', ',', 'D', '<', '>', 'v', ',', '8', '1', 'o', '*'])
        col = np.array(['r', 'lightgreen', 'b', 'y', 'purple', 'black', 'darkgoldenrod', 'magenta', 'darkgreen', 'darkblue',
                        'brown', 'darkgray', 'orange', 'dodgerblue', 'lightgray', 'cyan', ])
           
        with open(data_root_dir + "/tempdata/feature_names_all3.txt", 'r') as fh:
            lines = fh.readlines()
        
        featnames = lines[0].strip()
        featidxs = lines[1].strip()
        
        if featnames[-1] == ']':
            featnames = featnames[:-1]
        if featnames[0] == '[':
            featnames = featnames[1:]
            
        featidxs = np.fromstring(featidxs, dtype=int, sep=',') + nembeddings
        featnames = np.array(featnames.split(', '), dtype=str)
        
        for f, fname in enumerate(featnames):
            featnames[f] = featnames[f][2:] # skip the a1 bit at the start

            for catname in special_catnames:
                if catname == fname:
                    print("%i, Recognised %s as special cat %s" % (f, fname, catname))
                    feat_cats[nembeddings + f] = catname

            for catname in catnames:
                if catname in fname:
                    print("%i, Recognised %s as type %s" % (f, fname, catname))
                    feat_cats[nembeddings + f] = catname
                    break
            if not feat_cats[nembeddings + f]:
                print("%i, Unrecognised language feature: %s" % (f, fname))
                feat_cats[nembeddings + f] = 'ngram'


        for catname in catnames:
            print("No. features in category %s = %i" % (catname, np.sum(feat_cats == catname)))

        feat_cats[feat_cats=='_'] = 'ngram'

        # readability
        feat_cats[feat_cats=='ari'] = 'vocab/surface'
        feat_cats[feat_cats=='coleman'] = 'vocab/surface'
        feat_cats[feat_cats=='flesch'] = 'vocab/surface'

        feat_cats[feat_cats=='Rate'] = 'other'
        feat_cats[feat_cats=='Ratio'] = 'other'
        feat_cats[feat_cats=='Modal'] = 'other'
        feat_cats[feat_cats=='CONTEXTUALITY_MEASURE_FN'] = 'other'
        feat_cats[feat_cats == 'Ending'] = 'other'

        feat_cats[feat_cats=='_pos_ngram'] = 'POS'

        feat_cats[feat_cats=='_length'] = 'other'
        feat_cats[feat_cats=='word_more'] = 'other'
        feat_cats[feat_cats=='upperCaseRatio'] = 'other'
        feat_cats[feat_cats=='oovWordsCount'] = 'other'
        feat_cats[feat_cats=='spell_skill'] = 'other'
        feat_cats[feat_cats=='ExclamationRatio'] = 'other'

        feat_cats[feat_cats=='DependencyTreeDepth'] = 'other'
        feat_cats[feat_cats=='ProductionRule'] = 'prod. rule'

        feat_cats[feat_cats=='ner.type.'] = 'other'

        feat_cats[feat_cats=='sentiment'] = 'other'

        # for f in range(len(feat_cats)):
        #     feat_cats[f] = feat_cats[f].lower()

        print("After combining some categories.............................")

        for catname in np.unique(feat_cats):
            print("No. features in category %s = %i" % (catname, np.sum(feat_cats == catname)))

    # sort by length scale
    sorted_idxs = np.argsort(mean_ls)
    sorted_vals = mean_ls[sorted_idxs]
    
    # ignore those that were not valid
    sorted_vals = sorted_vals[totals[sorted_idxs]>0]
    sorted_idxs = sorted_idxs[totals[sorted_idxs]>0]

    sorted_cats = feat_cats[sorted_idxs]
    sorted_cats = sorted_cats[totals[sorted_idxs]>0]
    
    embeddingnames = np.empty(nembeddings, dtype=object)
    for e in range(nembeddings):
        embeddingnames[e] = 'Emb_dimension_%i' % e
        
    featnames = np.concatenate((embeddingnames, featnames))
    sorted_featnames = featnames[sorted_idxs]
    sorted_featnames = sorted_featnames[totals[sorted_idxs]>0]
    
    '''
    An alternative to plotting the distributions would be to list the top ten most important and least important features.
    '''
    figure_path = os.path.expanduser('./documents/pref_learning_for_convincingness/figures/features2/')
    
    np.savetxt(figure_path + '/feature_table.tex', np.concatenate((sorted_featnames[:, None], sorted_vals[:, None]), 
                                                                  axis=1), fmt='%s & %.5f \\nonumber\\\\')

    cat_arr = []
    labels = []
    for c, cat in enumerate(np.unique(feat_cats)):
        clengthscales = sorted_vals[sorted_cats == cat]
        cat_arr.append(clengthscales)
        labels.append(cat)

    # # Try a histogram instead? For each length-scale band, how many features of each type are there?
    # plt.figure()
    #
    # plt.hist(cat_arr, label=labels, color=col[:len(labels)], histtype='bar',
    #          bins=np.logspace(np.log10(1), np.log10(100000), 18), density=True) # density=True causes the values to be normalised
    # plt.xlabel('length-scale')
    # plt.ylabel('log_10 no. features')
    # plt.legend(loc='best')
    # plt.gca().set_xscale('log')
    #
    # plt.savefig(figure_path + 'hist.pdf')
    
    # produce content for a latex table

    matplotlib.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10,3))

    meds = []
    low = []
    high = []
    mins = []
    maxs = []
    vals = []
    for c, cat in enumerate(np.unique(feat_cats)):
        clengthscales = sorted_vals[sorted_cats == cat]
        #print '%s & %s & %s' & (cat, np.median(clengthscales), np.percentile(clengthscales, 25), np.percentile(clengthscales, 75))
        #meds.append(np.median(clengthscales))
        #low.append(np.percentile(clengthscales, 25))
        #high.append(np.percentile(clengthscales, 75))
        #mins.append(np.min(clengthscales))
        #maxs.append(np.max(clengthscales))
        vals.append(clengthscales)


        ax = plt.subplot(1, len(np.unique(feat_cats)), c+1)

        #plt.xlim(0, 20)
        plt.hist(clengthscales, label=labels[c], color='blue', histtype='bar',
                 #bins=np.logspace(np.log10(100), np.log10(100000), 24), density=False, orientation='horizontal')
                 #bins = np.logspace(np.log10(5500), np.log10(34000), 24), density = False, orientation = 'horizontal')
                 bins=np.arange(30) * 0.02 + 0.52, density=False, orientation='horizontal')

        # ax.set_yscale('log')
        #
        if c == 0:
             plt.ylabel('length-scale')# x10^3')
             #ax.get_yaxis().set_ticks([6e3, 1e4, 2e4, 3e4])
             #ax.get_yaxis().set_ticklabels(['6', '10', '20', '30'])
        else:
            ax.get_yaxis().set_ticks([])
            ax.get_yaxis().set_ticklabels([])

        #ax.get_xaxis().set_ticks([]) # write the x axis limits in the caption!!!
        plt.title(cat)

        #plt.gca().yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    # for i, v in enumerate(vals):
    #     vals[i] = np.log10(v)

    #bp = plt.boxplot(vals, labels=labels, notch=0, whiskerprops={'linestyle':'solid'},
    #                 patch_artist=True)
    #plt.setp(bp['boxes'], color='black')
    #plt.setp(bp['whiskers'], color='black')
    #for patch in bp['boxes']:
    #    patch.set_facecolor('tan')

    # yrange = np.arange(-2, 3)
    # plt.gca().set_yticks(yrange)
    # plt.gca().set_yticklabels(10.0**yrange)

    # plt.gca().set_axisbelow(True)

    #plt.ylim(0,3)

    plt.savefig(figure_path + 'boxplot.pdf')

    ############
    
    # plt.figure()
    #
    # rowsize = 5
    #
    # for c, cat in enumerate(np.unique(feat_cats)):
    #     clengthscales = sorted_vals[sorted_cats == cat]
    #     #plt.scatter(clengthscales, np.zeros(len(clengthscales)) + (1+c)*1000, marker=marks[c], color=col[c])
    #     ax = plt.subplot(len(labels)/rowsize + 1, rowsize, c+1)
    #     plt.plot(clengthscales, color=col[c], label=cat, marker=marks[c], linewidth=0)
    #     plt.title(cat)
    #     plt.ylim(np.min(sorted_vals), np.max(sorted_vals))
    #
    #     frame1 = plt.gca()
    #     if np.mod(c, rowsize):
    #         frame1.axes.get_yaxis().set_ticks([])
    #     else:
    #         plt.ylabel('length-scale')
    #     ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
    #
    # plt.xlabel('features')
    # plt.show()
    
    output = np.concatenate((sorted_cats[:, None], featnames[sorted_idxs][:, None], sorted_vals[:, None]), axis=1)
    np.savetxt("./results/feature_analysis/features.tsv", output, fmt='%s\t%s\t%s\t', delimiter='\t', header='category, feature_name, length-scale')

    # repeat this but make a separate sorted file by category
    for catname in np.unique(sorted_cats):
        catidxs = sorted_cats == catname
        output = np.concatenate((sorted_cats[catidxs, None], featnames[sorted_idxs][catidxs, None],
                                 sorted_vals[catidxs, None]), axis=1)
        np.savetxt("./results/feature_analysis/features_%s.tsv" % catname, output, fmt='%s\t%s\t%s\t', delimiter='\t',
                   header='category, feature_name, length-scale')


    print('all done.')