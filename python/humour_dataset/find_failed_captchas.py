#!/usr/bin/env python
# coding: utf-8

# Input: full_study.result.tsv, captcha_answers.tsv
# Output: assignmentid (column 18), workerID (column 19) for any assignment with a wrong captcha answer
# To check the captcha answers: 
# Load captcha IDs from captcha_answers.tsv, column 0, and answers from column 1
# Look in full_study.result.tsv at columns 31 - 51 (odd numbers) to get question IDs
# If any are captch questions, get their results from the preceding columns (even no.s)
# If there is a not a match, add the assignement ID and worker ID to the output data frame

import os, pandas as pd, numpy as np, sys
if len(sys.argv) > 1:
    if sys.argv[1] == '--help' or sys.argv[1] == '-h':
        print('Help: find_failed_captchas [path to captcha_answers and results.tsv]')
        sys.exit()
    path_to_data = sys.argv[1]
else:
    path_to_data = os.path.expanduser('~/Downloads/')
captchafile = os.path.join(path_to_data, 'captcha_answers.tsv')
resultfile = os.path.join(path_to_data, 'full_study.result.tsv')


captchaanswers = pd.read_csv(captchafile, sep='\t', index_col=0)

resultqids = pd.read_csv(resultfile, sep='\t', usecols=[30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50])

resultans = pd.read_csv(resultfile, sep='\t', usecols=[29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49])

# get the column ids in the results that contain the captchas
captchaidxs = np.argwhere(np.in1d(resultqids.values, captchaanswers.index.values).reshape(resultqids.shape))

# get the workers' captcha answers from the results
worker_answers = resultans.values[captchaidxs[:, 0], captchaidxs[:, 1]]

# get the ids of the captchas used at each result row
captcha_pairids = resultqids.values[captchaidxs[:, 0], captchaidxs[:, 1]]

# get the true answers
true_answers = captchaanswers.loc[captcha_pairids, 'CAPTCHA answer']

errors = (true_answers != worker_answers)
errors.shape

# get the assignment IDs and worker IDs for the error rows
result_ass_workers = pd.read_csv(resultfile, sep='\t', usecols=[18, 19])
bad_ass = result_ass_workers['assignmentid'].values[captchaidxs[:, 0]][errors.values]
bad_workers = result_ass_workers['workerid'].values[captchaidxs[:, 0]][errors.values]

# save to file
outfilename = os.path.join(path_to_data, 'failed_captchas.tsv')
output = pd.DataFrame({'assignmendid': bad_ass, 
                       'workerid': bad_workers})
output.to_csv(outfilename, sep='\t', index=False)
print('Saved failed captcha assignments and worker IDs to %s' % outfilename)



