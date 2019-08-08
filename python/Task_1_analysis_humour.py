
# coding: utf-8

# ### Compute results for task 1 on the humour dataset. 
# 
# To run this, you first need to produce the GPPL predictions by....
# 
# Then, set the variable resfile to point to the ouput folder of the previous step. 
# 
# TODO:
#    * see metaphor variant to work out which bits of this we actually need to keep
#    * Loook at the files under bws_vs_gppl/python and in run_experiments.py to work out how to run GPPL. 

# In[1]:


get_ipython().run_line_magic('pylab', '')
get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
import pandas as pd
import os, logging, csv
from nltk.tokenize import word_tokenize
from scipy.stats.mstats import spearmanr, pearsonr


# In[4]:


# Where to find the predictions and gold standard
resfile = os.path.expanduser('./results/experiment_humour_2019-02-28_16-39-36/results-2019-02-28_16-39-36.csv')
                             #experiment_humour_2019-02-26_20-44-52/results-2019-02-26_20-44-52.csv')


# In[22]:


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

# In[ ]:


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

# In[23]:


# This cell compares the predicted ratings to the gold standard *classifications* to see if the ratings can be used
# to separate funny and non-funny.

# load the discrete labels
catfile = os.path.expanduser('./results/experiment_humour_2019-02-28_16-39-36/cats/results-2019-02-28_20-45-25.csv')
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


# In[8]:


# Defines a function for loading the humour data.

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


# In[12]:


# Load the comparison data provided by the crowd
datafile = os.path.expanduser('../data/pl-humor-full/results.tsv')

pairs, idxs = load_crowd_data_TM(datafile)

pairs = np.array(pairs)
np.savetxt(os.path.expanduser('../data/pl-humor-full/pairs.csv'), pairs, '%i', delimiter=',')


# In[26]:


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
# 

# In[28]:


# Table 3: For the humour dataset, compute the correlation between the gold standard and the BWS scores with subsets of data.

# take random subsets of pairs so that each pair has only 4 annotations
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


# In[29]:


# computer Krippendorff's alpha agreement score.

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

# In[ ]:


from scipy.stats import rankdata

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

# In[ ]:


all_comp = []
all_comp_gppl = []
all_worst_loss = []
all_best_beaten = []

# Do diffs correlate with sum(- worse_item_rank + better_item_rank)?
for idx in range(len(diffs)):
    #print('Item: %i' % ids[idx])
    #print('Diff: %f; BWS rank=%i, GPPL rank=%i' % (diffs[idx], rank_bws[idx], rank_gppl[idx]))
    
    otherids = pairs[pairs[:, 0] == ids[idx], 1]
    otheridxs = [np.argwhere(ids == otherid).flatten()[0] for otherid in otherids]
    
    tot_rank_gppl = 0
    tot_rank_bws = 0
    
    best_beaten = 20000    
    for otheridx in otheridxs:
        tot_rank_gppl -= rank_gppl[otheridx]
        tot_rank_bws -= rank_bws[otheridx]
        
        if rank_bws[otheridx] < best_beaten:
            best_beaten = rank_bws[otheridx]
        
    worst_loss = 0
        
    otherids = pairs[pairs[:, 1] == ids[idx], 0]
    otheridxs = [np.argwhere(ids == otherid).flatten()[0] for otherid in otherids]
    for otheridx in otheridxs:
        tot_rank_gppl += rank_gppl[otheridx]
        tot_rank_bws += rank_bws[otheridx]
        
        if rank_bws[otheridx] > worst_loss:
            worst_loss = rank_bws[otheridx]        
            
    #print('Total rank differences: BWS=%i, GPPL=%i' % (tot_rank_gppl, tot_rank_bws))
    all_comp.append(tot_rank_bws)
    all_comp_gppl.append(tot_rank_gppl)
    all_worst_loss.append(worst_loss)
    all_best_beaten.append(best_beaten)
print('Correlation between rank diff and total ranks of compared items: %f' % spearmanr(all_comp, diffs)[0])
print('Correlation between rank diff and total ranks of compared items: %f' % spearmanr(all_comp_gppl, diffs)[0])
print('Correlation between rank diff and the best item that this item beat: %f' % spearmanr(all_best_beaten, diffs)[0])
print('Correlation between rank diff and the worst item that beat this item: %f' % spearmanr(all_worst_loss, diffs)[0])


# In[ ]:


print(pearsonr(all_comp, diffs))
print(pearsonr(all_comp_gppl, diffs))


# In[ ]:


counts = np.array(counts)
tmp = deduped_pairs[counts<0, 0]
deduped_pairs[counts < 0, 0] = deduped_pairs[counts < 0, 1]
deduped_pairs[counts < 0, 1] = tmp
deduped_pairs = deduped_pairs.astype(int)
#print(deduped_pairs)


# In[ ]:


# Do the discrepancies correspond with worker disagreements (same pairs with multiple values)?
# Counts close to 0 indicate lots of disagreemnts
# What is the average count for the de-duplicated pairs for (a) items with > 1000 rank discrepancy 
# and (b) the other items?
# Is the difference statistically significant?

big_diff_mean_counts = []
small_diff_mean_counts = []

for i, itemid in enumerate(ids):
    # get all deduped pairs for this id
    pairidxs = np.argwhere((deduped_pairs[:, 0] == itemid) | (deduped_pairs[:, 1] == itemid))
    
    mean_vote_counts = np.mean(np.abs(counts[pairidxs]))
    
    if i in bigdiffidxs:
        big_diff_mean_counts.append(mean_vote_counts)
    else:
        small_diff_mean_counts.append(mean_vote_counts)


# In[ ]:


print(np.mean(big_diff_mean_counts))
print(np.mean(small_diff_mean_counts))


# In[ ]:


print(np.var(big_diff_mean_counts))
print(np.var(small_diff_mean_counts))

