'''
Created on Dec 21, 2017

@author: simpson
'''
from svmutil import svm_read_problem, svm_train, svm_predict
from scipy.stats.stats import pearsonr
import numpy as np
from sklearn.svm.classes import NuSVR
from sklearn.datasets.svmlight_format import load_svmlight_file

if __name__ == '__main__':
        
    trainfile = './data/svm_train.txt'
        
    problem = svm_read_problem(trainfile)
    rank_model = svm_train(problem[0][:-100], problem[1][:-100], '-s 4 -h 0 -m 1000')
 
    predicted_f, _, _ = svm_predict(np.ones(100).tolist(), problem[1][-100:], rank_model)

    scores_rank_test = problem[0][-100:]

    print(("Pearson correlation for fold = %f" % pearsonr(scores_rank_test, predicted_f)[0]))
    
    svr = NuSVR()
    
    lingfeat, y = load_svmlight_file(trainfile)
    
    svr.fit(lingfeat[:-100], y[:-100])
    y_pred = svr.predict(lingfeat[-100:])
    
    print(("Pearson correlation for fold = %f" % pearsonr(scores_rank_test, y_pred)[0]))
    