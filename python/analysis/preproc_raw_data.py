'''
Created on 10 May 2016

@author: simpson
'''
import xmltodict, os
import numpy as np
import logging

datadir = '../../data/outputdata'
exptlabel = '3'
plotdir = './results%s/' % exptlabel

def load(data, f):
    #logging.warning("Subsampling dataset for debugging!!!")
    #data = data[:1000, :]
    
    npairs = data.shape[0]
    
    arg_ids = np.unique([data[:, 1], data[:, 2]])
    ntexts = np.max(arg_ids) + 1 # +1 because there can also be an argument with ID 0
    
    pair1idxs = data[:, 1].astype(int)
    pair2idxs = data[:, 2].astype(int)
    prefs = data[:, 3].astype(float) / 2.0 # the labels give the preference for argument 2. Halve the 
    #values so they sit between 0 and 1 inclusive. Labels expressing equal preference will be 0.5.

    #The feature coordinates for the arguments.
    xvals = np.arange(ntexts) # ignore the argument features for now, use indices so that
    # the different arguments are not placed at the exact same location -- allows for ordering of the arguments. We 
    # have to combine this with a diagonal covariance function.
    yvals = np.zeros(ntexts)
    
    logging.info( "Testing Bayesian preference components analysis using real crowdsourced data...")
    
    nx = 1
    ny = 1
    
    pair1coords = np.concatenate((xvals[pair1idxs][:, np.newaxis], yvals[pair1idxs][:, np.newaxis]), axis=1)
    pair2coords = np.concatenate((xvals[pair2idxs][:, np.newaxis], yvals[pair2idxs][:, np.newaxis]), axis=1) 

    personids = data[:, 0].astype(int)
    upersonids = np.unique(personids)
    nworkers = len(upersonids)
    
    return datadir, plotdir, nx, ny, data, pair1coords, pair2coords, pair1idxs, pair2idxs, xvals, yvals, prefs, \
            personids, npairs, nworkers, ntexts, f
    
def load_synthetic(acc=0.9):
    # Random data generation
    npairs = 10000
    nitems = 5
    nclusters = 2 # true number of clusters
    f = np.random.rand(nclusters, nitems) * 10
    
    nworkers = 10
    clusterids = np.random.randint(0, nclusters, (nworkers))    
    
    data0 = np.random.randint(0, nworkers, (npairs,1)) # worker ids
    data1 = np.random.randint(0, nitems, (npairs,1)) # pair 1 ids
    data2 = np.random.randint(0, nitems, (npairs,1)) # pair 2 ids
    
    # remove duplicates. The methods should cope with these but they might distort results
#     for d in range(npairs):
#         dupes = np.in1d(data0[d], data0[:d]) & np.in1d(data1[d], data1[:d]) & np.in1d(data2[d], data2[:d])
#         while np.any(dupes):
#             data0[d] = np.random.randint(0, nworkers, 1) # worker ids
#             data1[d] = np.random.randint(0, nitems, 1) # pair 1 ids
#             data2[d] = np.random.randint(0, nitems, 1) # pair 2 ids   
#             dupes = np.in1d(data0[d], data0[:d]) & np.in1d(data1[d], data1[:d]) & np.in1d(data2[d], data2[:d])         
    
    # the function values for the first items in the pair, given the cluster id of the annotator 
    f1 = f[clusterids[data0], data1]
    # the function values for the second items in the pair, given the cluster id of the annotator 
    f2 = f[clusterids[data0], data2]
        
    correctflag = np.random.rand(npairs, 1) # use this to introduce noise into the preferences instead of reflecting f precisely
    #if f1 < f2 by more than 0.5 then the answer is 2
    data3_f2greater = 2 * (correctflag < acc) * (f1+0.5 < f2) 
    #if f1 = f2 to within 0.5 then the answer is 1
    data3_f1f2same =  1 * (correctflag < acc) * (np.abs(f1 - f2) <=0.5)  
    #if the worker makes an error, it is random
    data3_incorrect = (correctflag > acc) * np.random.randint(0, 3, (npairs, 1))
    # combine the possible labels
    data3 = data3_f2greater + data3_f1f2same + data3_incorrect
    logging.debug('Number of neg prefs = %i, no prefs = %i, pos prefs = %i' % (np.sum(data3==0), np.sum(data3==1), np.sum(data3==2)))
    data = np.concatenate((data0, data1, data2, data3), axis=1)
    
    return load(data, f)
    
def load_amt():
    # Task A1 ---------------------------------------------------------------------------------------------------------
    # load the data with columns: person_ID, arg_1_ID, arg_2_ID, preference_label
    data = np.genfromtxt(datadir + '/all_labels.csv', dtype=int, delimiter=',') 
    f = []    
    return load(data, f)

def process_list(pairlist):
    
    crowdlabels = []
    
    pairlist = pairlist['annotatedArgumentPair']
    for pair in pairlist:
        for workerlabel in pair['mTurkAssignments']['mTurkAssignment']:
            row = [workerlabel['turkID'], pair['arg1']['id'], pair['arg2']['id'], workerlabel['value']]
            crowdlabels.append(row)
        
    return np.array(crowdlabels)

def process_list_of_turker_assts(pairlist):
    
    crowdlabels = []
    
    pairlist = pairlist['annotatedArgumentPair']
    for pair in pairlist:
        workerlabel = pair['mTurkAssignments']['mTurkAssignment']
        row = [pair['id'].encode('utf-8'), workerlabel['turkID'].encode('utf-8'), 
               workerlabel['value'].encode('utf-8'),
               pair['arg1']['text'].encode('utf-8').replace('\n', ' ').replace('\t', ' '), 
               pair['arg2']['text'].encode('utf-8').replace('\n', ' ').replace('\t', ' ')]
        crowdlabels.append(row)
        
    return np.array(crowdlabels)

def process_list_of_gold(pairlist):
    
    crowdlabels = []
    
    pairlist = pairlist['annotatedArgumentPair']
    for pair in pairlist:
        if 'goldLabel' in pair:            
            row = [pair['id'].encode('ascii','ignore'), 'goldLabel',
                   pair['goldLabel'].encode('ascii','ignore'),
                   pair['arg1']['text'].replace('\n', ' ').replace('\t', ' ').encode('ascii','ignore'),
                   pair['arg2']['text'].replace('\n', ' ').replace('\t', ' ').encode('ascii','ignore')]
            crowdlabels.append(row)
        
    return np.array(crowdlabels)

def translate_to_local(all_labels):
    _, localworkers = np.unique(all_labels[:, 0], return_inverse=True)
    _, localargs = np.unique([all_labels[:, 1], all_labels[:, 2]], return_inverse=True)
    localargs1 = localargs[0:all_labels.shape[0]]
    localargs2 = localargs[all_labels.shape[0]:]
    ulabels = {'a1':0, 'a2':2, 'equal':1}
    locallabels = [ulabels[l] for l in all_labels[:, 3]]
    
    all_labels = np.zeros(all_labels.shape)
    all_labels[:, 0] = localworkers
    all_labels[:, 1] = localargs1
    all_labels[:, 2] = localargs2
    all_labels[:, 3] = locallabels
    return all_labels

def generate_gold_CSV(datadir, outputdir):
    datafiles = os.listdir(datadir)
    for i, f in enumerate(datafiles):
        if f.split('.')[-1] != 'xml':
            continue        
        print("Processing file %i of %i, filename=%s" % (i, len(datafiles), f))
        with open(datadir + f, 'r') as ffile:
            doc = xmltodict.parse(ffile.read())
            pairlist = doc['list']
            labels = process_list_of_gold(pairlist)    
            np.savetxt(outputdir + '/' + f.split('.')[0] + '.csv', labels, fmt='%s\t%s\t%s\t%s\t%s', delimiter='\t', 
                       header='#id    annotatorid    label    a1    a2', )

def generate_turker_CSV(datadir, outputdir):
    datafiles = os.listdir(datadir)
    for i, f in enumerate(datafiles):
        if f.split('.')[-1] != 'xml':
            continue        
        print("Processing file %i of %i, filename=%s" % (i, len(datafiles), f))
        with open(datadir + f) as ffile:
            doc = xmltodict.parse(ffile.read())
            pairlist = doc['list']
            labels = process_list_of_turker_assts(pairlist)    
            np.savetxt(outputdir + '/' + f.split('.')[0] + '.csv', labels, fmt='%s\t%s\t%s\t%s\t%s', delimiter='\t', 
                       header='#id    turkerid    label    a1    a2', )          

if __name__ == '__main__':
    '''
    Read in the original XML file. Extract each element as line in the CSV file with IDs of the text blocks, the user,
    the value they assigned.
    '''
    datadir = '/home/local/UKP/simpson/data/step5-gold-data-all/'
    datafiles = os.listdir(datadir)
    #datafiles = ["testonly.xml"] # try with one file first
    
    outputdir = '../../data/outputdata'
    
    all_labels = np.empty((0, 4))
    
    for i, f in enumerate(datafiles):
        print("Processing file %i of %i, filename=%s" % (i, len(datafiles), f))
        with open(datadir + f) as ffile:
            doc = xmltodict.parse(ffile.read())
            pairlist = doc['list']
            labels = process_list(pairlist)            
            all_labels = np.concatenate((all_labels, labels), axis=0)
            
    np.savetxt(outputdir + '/all_labels_original.csv', all_labels, fmt='%s, %s, %s, %s', delimiter=',')
    all_labels = translate_to_local(all_labels)
    np.savetxt(outputdir + '/all_labels.csv', all_labels, fmt='%i, %i, %i, %i', delimiter=',')