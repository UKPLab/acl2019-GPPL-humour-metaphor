
# coding: utf-8

# ### Compute results for task 1 on the humour dataset. 
# 
# Please see the readme for instructions on how to produce the GPPL predictions that are required for running this script.
# 
# Then, set the variable resfile to point to the ouput folder of the previous step. 
#
import string

import pandas as pd
import os, logging, csv
from nltk.tokenize import word_tokenize
from scipy.stats.mstats import spearmanr, pearsonr
import numpy as np

# Where to find the predictions and gold standard
resfile = './results/experiment_humour_2019-02-26_20-44-52/results-2019-02-26_20-44-52.csv'

# Load the data
data = pd.read_csv(resfile, usecols=[0,1,2])
ids = data['id'].values
bws = data['bws'].values
gppl = data['predicted'].values

# ### Ties in the BWS Scores contribute to the discrepeancies between BWS and GPPL
# 
# GPPL scores are all unique, but BWS contains many ties. 
# Selecting only one of the tied items increases the Spearman correlation.
# 
# Find the ties in BWS. Compute correlations between those tied items for the GPPL scores vs. original BWS scores and GPPL vs. scaled BWS scores.
# Do the ties contribute a lot of the differences in the overall ranking?
# Another way to test if the ties contribute differences to the ranking:
# Select only one random item from each tie and exclude the rest, then recompute. 
print('with ties included:')
print(spearmanr(bws, gppl)[0])
print('with ties present but no correction for ties:')
print(spearmanr(bws, gppl, False)[0])
print('with a random sample of one item if there is a tie in bws scores:')
total = 0
for sample in range(10):
    untied_sample_bws = []
    untied_sample_gppl = []
    
    ties = []
    tiesgppl = []
    
    for i, item in enumerate(ids):

        if i >= 1 and bws[i] == bws[i-1]:
            
            if len(ties) == 0 or i-1 != ties[-1]:
                ties.append(i-1) # the previous one should be added to the list if we have just recognised it as a tie
            
            ties.append(i)
            #randomly choose whether to keep the previous item or this one
            if np.random.rand() < 0.5:
                pass
            else:
                untied_sample_bws.pop()
                untied_sample_gppl.pop()

                untied_sample_bws.append(bws[i])
                untied_sample_gppl.append(gppl[i])
        else:
            untied_sample_bws.append(bws[i])
            untied_sample_gppl.append(gppl[i])
        
        if i >= 1 and gppl[i] == gppl[i-1]:
            
            if len(tiesgppl) == 0 or i-1 != tiesgppl[-1]:
                tiesgppl.append(i-1) # the previous one should be added to the list if we have just recognised it as a tie
            
            tiesgppl.append(i)
            
    rho = spearmanr(untied_sample_bws, untied_sample_gppl)[0]
    total += rho
    print(rho)
    
    print('Number of BWS tied items = %i' % len(ties))
    print('Number of GPPL tied items = %i' % len(tiesgppl))
    
    sample_size = len(untied_sample_bws)
    
print('Mean for samples without ties = %f' % (total / 10))
print('Correlations for random samples of the same size (%i), allowing ties: ' % sample_size)
total = 0
for sample in range(10):
    
    # take a random sample, without caring about ties
    randidxs = np.random.choice(len(bws), sample_size, replace=False)
    rho = spearmanr(bws[randidxs], gppl[randidxs])[0]
    print(rho)
    total += rho
    
print('Mean rho for random samples = %f' % (total / 10))


# ### Hypothesis: the ratings produced by BWS and GPPL can be used to separate the funny from non-funny sentences.

# This compares the predicted ratings to the gold standard *classifications* to see if the ratings can be used
# to separate funny and non-funny.

# load the discrete labels

def get_cats(fname):
    with open(os.path.join('./data/pl-humor-full', fname), 'r') as f:
        for line in f:
            line = line.strip()
            for c in string.punctuation + ' ' + '\xa0':
                line = line.replace(c, '')
#            line = line.replace(' ', '').strip()
#            line = line.replace('"', '')  # this is probably borked by tokenization?
            instances[line] = cats[fname]


def assign_cats(fname):
    with open(fname, 'r') as fr, open(fname + '_cats.csv', 'w') as fw:
        reader = csv.DictReader(fr)
        writer = csv.DictWriter(fw, fieldnames=['id', 'bws', 'predicted', 'category', 'sentence'])
        writer.writeheader()
        for row in reader:
            sentence = row['sentence'].strip()
            for c in string.punctuation + ' ':
                sentence = sentence.replace(c, '')
#            sentence = row['sentence'].replace(' ','').strip()
#            sentence = sentence.replace('`', '\'')  # this is probably borked by tokenization?
#            sentence = sentence.replace('"', '')  # this is probably borked by tokenization?
            row['category'] = instances[sentence]
            writer.writerow(row)

cats = dict()
cats['jokes_heterographic_puns.txt'] = 'hetpun'
cats['jokes_homographic_puns.txt'] = 'hompun'
cats['jokes_nonpuns.txt'] = 'nonpun'
cats['nonjokes.txt'] = 'non'

instances = dict()

for fname in cats.keys():
    get_cats(fname)

assign_cats(resfile)

catfile = os.path.expanduser(resfile + '_cats.csv')
#'./results/experiment_humour_2019-02-28_16-39-36/cats/results-2019-02-28_20-45-25.csv')
cats = pd.read_csv(catfile, index_col=0, usecols=[0,3])

cat_list = np.array([cats.loc[instance].values[0] if instance in cats.index else 'unknown' for instance in ids])

gfunny = (cat_list == 'hompun') | (cat_list == 'hetpun')
gunfunny = (cat_list == 'nonpun') | (cat_list == 'non')

print('Number of funny = %i, non-funny = %i' % (np.sum(gfunny), 
                                                np.sum(gunfunny) ) )

# check classification accuracy  --  how well does our ranking separate the two classes
from sklearn.metrics import roc_auc_score

gold = np.zeros(len(cat_list))
gold[gfunny] = 1
gold[gunfunny] = 0
goldidxs = gfunny | gunfunny
gold = gold[goldidxs]

print('AUC for BWS = %f' % roc_auc_score(gold, bws[goldidxs]) )
print('AUC for GPPL = %f' % roc_auc_score(gold, gppl[goldidxs]) )

# a function for loading the humour data.
def load_crowd_data_TM(path):
    """
    Read csv and create preference pairs of tokenized sentences.

    :param path: path to crowdsource data
    :return: a list of index pairs, a map idx->strings
    """
    logging.info('Loading crowd data...')

    pairs = []
    idx_instance_list = []
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # skip header row
        for line_no, line in enumerate(reader):
            answer = line[1]
            A = word_tokenize(line[2])
            B = word_tokenize(line[3])
            # add instances to list (if not alreay in it)
            if A not in idx_instance_list:
                idx_instance_list.append(A)
            if B not in idx_instance_list:
                idx_instance_list.append(B)
            # add pairs to list (in decreasing preference order)
            if answer == 'A':
                pairs.append((idx_instance_list.index(A), idx_instance_list.index(B)))
            if answer == 'B':
                pairs.append((idx_instance_list.index(B), idx_instance_list.index(A)))
    return pairs, idx_instance_list

# Load the comparison data provided by the crowd
datafile = os.path.expanduser('./data/pl-humor-full/results.tsv')

pairs, idxs = load_crowd_data_TM(datafile)

pairs = np.array(pairs)
np.savetxt(os.path.expanduser('./data/pl-humor-full/pairs.csv'), pairs, '%i', delimiter=',')

# For each item compute its BWS scores
# but scale by the BWS scores of the items they are compared against.
# This should indicate whether two items with same BWS score should 
# actually be ranked differently according to what they were compared against.
def compute_bws(pairs):
    new_bws = []
    for i, item in enumerate(ids):

        matches_a = pairs[:, 0] == item
        matches_b = pairs[:, 1] == item

        new_bws.append((np.sum(matches_a) - np.sum(matches_b)) 
                       / float(np.sum(matches_a) + np.sum(matches_b)))
        
    return new_bws


# ### Agreement and consistency of annotators

# Table 3: For the humour dataset, compute the correlation between the gold standard and the BWS scores with subsets of data.

# Take random subsets of pairs so that each pair has only 4 annotations
def get_pid(pair):
    return '#'.join([str(i) for i in sorted(pair)])

def compute_mean_correlation(nannos):
    nreps = 10

    mean_rho = 0

    for rep in range(nreps):
        pair_ids = list([get_pid(pair) for pair in pairs])
        upair_ids = np.unique(pair_ids)
        anno_counts = np.zeros(len(upair_ids))
        subsample = []
        for p, pid in enumerate(np.random.choice(pair_ids, len(pair_ids), replace=False)):

            if anno_counts[upair_ids == pid] < nannos:
                anno_counts[upair_ids == pid] += 1
                subsample.append(p)
        print('Got subsample')
        sub_pairs = pairs[subsample]
        sub_bws = compute_bws(sub_pairs)
        # Now compute the correlations again
        mean_rho += spearmanr(bws, sub_bws)[0]

    mean_rho /= nreps
    print('Mean rho for %i = %f' % (nannos, mean_rho))
    
for nannos in range(1, 5):
    compute_mean_correlation(nannos)

# Compute Krippendorff's alpha agreement score.
def alpha(U, C, L):
    '''
    U - units of analysis, i.e. the data points being labelled
    C - a list of classification labels
    L - a list of labeller IDs
    '''
    N = float(np.unique(U).shape[0])
    Uids = np.unique(U)
    print(Uids)
    
    Dobs = 0.0
    Dexpec = 0.0
    for i, u in enumerate(Uids):
        uidxs = U==u
        
        Lu = L[uidxs]
        m_u = Lu.shape[0]
        
        if m_u < 2:
            continue
        
        Cu = C[uidxs]
        
        #for cuj in Cu:
        #    Dobs += 1.0 / (m_u - 1.0) * np.sum(np.abs(cuj - Cu))
        
        Dobs += 1.0 / (m_u - 1.0) * np.sum(np.abs(Cu[:, np.newaxis] - Cu[np.newaxis, :]) != 0)

    # too much memory required            
    # Dexpec = np.sum(np.abs(C.flatten()[:, np.newaxis] - C.flatten()[np.newaxis, :]))
            
    for i in range(len(U)):
        if np.sum(U==U[i]) < 2:
            continue
        Dexpec += np.sum(np.abs(C[i] - C) != 0) # sum up all differences regardless of user and data unit
        
    Dobs = 1 / N * Dobs
    Dexpec = Dexpec / (N * (N-1))  
    
    alpha = 1 - Dobs / Dexpec
    return alpha

data = pd.read_csv(datafile, usecols=[0, 1], sep='\t')
print(data.columns)
L = data.loc[data['Answer'] != 'X']['Worker ID'].values
print(L.shape)
C = data.loc[data['Answer'] != 'X']['Answer'].values
C[C == 'A'] = 0
C[C == 'B'] = 1
U = np.array([get_pid(pair) for pair in pairs])
print(len(U))

alpha(U, C, L)

# ### The ranking discrepancies are mostly very small
# 
# The plot below shows that the distribution is very small.
# 
# However, some items are very distantly ranked -- we will investigate this in the following cells.
from scipy.stats import rankdata
import matplotlib.pyplot as plt

rank_bws = rankdata(-bws)
rank_gppl = rankdata(-gppl)

diffs = rank_bws - rank_gppl

plt.figure(figsize=(2.1, 2))
plt.hist(diffs, bins=51)
plt.xlabel('Rank difference ') # (BWS rank - GPPL rank) ==> put in the caption
plt.ylabel('No. sentences')
plt.tight_layout()

plt.savefig(os.path.expanduser('./results/humor_rank_diff_hist.pdf'))

# ### Reasons for discrepancies: Weights of compared items
# 
# GPPL ranks some instances lower because the items they lost against were much lower-ranked?
# 
# GPPL considers the weights of items that each item is compared against. 
# Is there a correlation between the total rank of instances that a given instance is compared against, 
# and the difference between BWS and GPPL scores?
# 
all_comp_gppl = []

# Do diffs correlate with sum(- worse_item_rank + better_item_rank)?
for idx in range(len(diffs)):
    #print('Item: %i' % ids[idx])
    #print('Diff: %f; BWS rank=%i, GPPL rank=%i' % (diffs[idx], rank_bws[idx], rank_gppl[idx]))
    
    otherids = pairs[pairs[:, 0] == ids[idx], 1]
    otheridxs = [np.argwhere(ids == otherid).flatten()[0] for otherid in otherids]
    
    tot_rank_gppl = 0

    for otheridx in otheridxs:
        tot_rank_gppl -= rank_gppl[otheridx]


    otherids = pairs[pairs[:, 1] == ids[idx], 0]
    otheridxs = [np.argwhere(ids == otherid).flatten()[0] for otherid in otherids]
    for otheridx in otheridxs:
        tot_rank_gppl += rank_gppl[otheridx]

    #print('Total rank differences: BWS=%i, GPPL=%i' % (tot_rank_gppl, tot_rank_bws))
    all_comp_gppl.append(tot_rank_gppl)
print('Correlation between rank diff and total ranks of compared items: %f' % spearmanr(all_comp_gppl, diffs)[0])
print(pearsonr(all_comp_gppl, diffs))

