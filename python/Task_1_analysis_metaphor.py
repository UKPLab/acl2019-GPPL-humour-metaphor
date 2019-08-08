# coding: utf-8

# ### Task 1: computing correlations between GPPL and BWS when no features are used, but all pairwise labels are used.
# 
# To run, please see the readme for instructions on how to produce the GPPL scores.
# 
# Then, set the resfile variable below to point to the results of the previous step.

import pandas as pd
import os, logging, csv
from scipy.stats.mstats import spearmanr, pearsonr
import numpy as np

# resfile = os.path.expanduser('./results/experiment_metaphor_2019-02-28_16-42-11/results-2019-02-28_22-54-46.csv')
# resfile = os.path.expanduser('./results/debug_metaphor_2019-02-28_13-05-50/results-2019-02-28_13-05-50.csv')
resfile = './results/debug_metaphor_2019-02-26_17-32-31/results-2019-02-26_17-32-31.csv'

# Load the data
data = pd.read_csv(resfile, usecols=[0,1,2])
ids = data['id'].values
bws = data['novelty'].values
gppl = data['predicted'].values

# ### Hypothesis: Ties in the BWS Scores contribute to the discrepeancies between BWS and GPPL.
# 
# GPPL scores are all unique, but BWS contains many ties. Selecting only one of the tied items increases the Spearman correlation.

# Compute the task 1 results for table 5.

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


# ### The ranking discrepancies are mostly very small
# 
# The plot below shows that the distribution is very small.
# 
# However, some items are very distantly ranked -- we will investigate this in the following cells.

# produce the plot for figure 1 showing distribution of rank differences between BWS and GPPL.
from scipy.stats import rankdata
import matplotlib.pyplot as plt

rank_bws = rankdata(-bws)
rank_gppl = rankdata(-gppl)

diffs = rank_bws - rank_gppl

plt.figure(figsize=(2, 2))
plt.hist(diffs, bins=51)
plt.xlabel('Rank difference ') # put in the caption: (BWS rank - GPPL rank)
#plt.ylabel('Number of sentences') # don't need this
plt.tight_layout()

plt.savefig(os.path.expanduser('./results/metaphor_rank_diff_hist.pdf'))

from collections import namedtuple, OrderedDict
import re

def load_crowd_data_ED(path):
    """
    Read csv and create preference pairs of VUAMC ids representing sentences with focus.

    :param path: path to crowdsource data
    :return: a list of index pairs, a map idx->vuamc_id
    """
    logging.info('Loading crowd data...')

    pairs = []
    idx_instance_list = []

    skipped = 0
    corrected = [0]
    
    total_annos = 0
    
    # create 3*5 pairs from each line:
    # - +3: a = best > b, c, d
    # - +2: d = worst < b, c
    # - *3, because a HIT contains 3 comparisons
    Hit = namedtuple('Hit', 'ids nov_id con_id')

    def correct(id_map, bws_id):
        try:
            if bws_id.startswith('$'):
                # print('  Corrected bws id', bws_id)
                corrected[0] += 1
                return id_map[bws_id[1:]]
            return id_map[bws_id]
        except KeyError:
            return ''

    if os.path.isfile(path):
        paths = [path]
    elif os.path.isdir(path):
        paths = [os.path.join(path, fname) for fname in os.listdir(path)]

    for path in paths:
      logging.info('Loading crowd data from %s...', path)
      with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for line_no, line in enumerate(reader):
            for i in range(0,3):  # 3 assignments per HIT
                # add instances to list (if not alreay in it)
                bws_vuamc_ids = OrderedDict(zip(line[27+(i*16):27+(i*16) + 4], line[27+(i*16)+4:27+(i*16) + 4 + 4]))
                con_id = bws_vuamc_ids[line[87+(i*2)]] if line[87+(i*2)] in bws_vuamc_ids else None
                nov_id = bws_vuamc_ids[line[88+(i*2)]] if line[88+(i*2)] in bws_vuamc_ids else None
                # check hit validity
                
                total_annos += 1
                
                if nov_id is None or con_id is None or None in [re.match(r'^\d+\.\d+\.\d+$',vuamc_id) for vuamc_id in bws_vuamc_ids.values()]:
#                    logging.warn('Skipping corrupt VUAMC id in line [%s]', line_no)
                    continue
                for vuamc_id in bws_vuamc_ids.values():
                    if vuamc_id not in idx_instance_list:
                        idx_instance_list.append(vuamc_id)
                if OPTIONS['data_mode'] == 'pairs':
                    pairs.append((idx_instance_list.index(nov_id), idx_instance_list.index(con_id)))
                else:
                    for current_id in bws_vuamc_ids.values():
                        if current_id != nov_id:  # best (a) > b, c, d
                            pairs.append((idx_instance_list.index(nov_id), idx_instance_list.index(current_id)))  # append "novel" pairs
                            if current_id != con_id:  # worst (d) < b, c
                                pairs.append((idx_instance_list.index(current_id), idx_instance_list.index(con_id)))  # append "conventionalized" pairs
                                
                                
    print('Total annotations: %i' % total_annos)
    return pairs, idx_instance_list

# Load the comparison data
OPTIONS = {}
OPTIONS['data_mode'] = 'pairs'

pairs, idx_instance_list = load_crowd_data_ED(os.path.expanduser('./data/vuamc_crowd/all.csv'))
#np.savetxt(os.path.expanduser('./data/pairs.csv'), pairs, '%i', delimiter=',')

pairs = np.array(pairs)
idxs = np.arange(len(idx_instance_list))
upairids = np.unique(pairs)
uidxs = np.unique(idxs)
np.all(np.in1d(upairids, uidxs))

# ### Weights of compared items
# 
# GPPL considers the weights of items that each item is compared against. 
# Is there a correlation between the total rank of instances that a given instance is compared against, 
# and the difference between BWS and GPPL scores?

all_comp = []
all_comp_gppl = []
all_worst_loss = []
all_best_beaten = []

# Do diffs correlate with sum(- worse_item_rank + better_item_rank)?
for idx in range(len(diffs)):
    #print('Item: %s' % ids[idx])
    #print('Diff: %f; BWS rank=%i, GPPL rank=%i' % (diffs[idx], rank_bws[idx], rank_gppl[idx]))
    
    otherids = pairs[pairs[:, 0] == idx, 1]
    otheridxs = [np.argwhere(idxs == otherid).flatten()[0] for otherid in otherids]
    
    tot_rank_gppl = 0
    for otheridx in otheridxs:
        tot_rank_gppl -= rank_gppl[otheridx]
                
    otherids = pairs[pairs[:, 1] == idx, 0]
    otheridxs = [np.argwhere(idxs == otherid).flatten()[0] for otherid in otherids]
    for otheridx in otheridxs:
        tot_rank_gppl += rank_gppl[otheridx]      
            
    #print('Total rank differences: BWS=%i, GPPL=%i' % (tot_rank_gppl, tot_rank_bws))
    all_comp_gppl.append(tot_rank_gppl)

print('Spearman correlation between rank diff and total ranks of compared items: %f' % spearmanr(all_comp_gppl, diffs)[0])

# pearson correlation for the above.
print('Pearson correlation for same: %f' % pearsonr(all_comp_gppl, diffs)[0])

