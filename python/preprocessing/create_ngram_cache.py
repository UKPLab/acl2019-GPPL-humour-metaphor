#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import logging
import nltk
import os
import re
import string
import sys
from collections import defaultdict
from vuamc import Vuamc

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# n
n = 2

# path to data file
humour_path = '/home/dodinh/pl-humor-full/results.tsv'
vuamc_path = '/home/dodinh/data/VU Amsterdam Metaphor Corpus/2541/VUAMC_with_novelty_scores.xml'
output_path_tmpl = '/home/dodinh/data/ngrams/{}_all_{}grams.csv2'
error_path_tmpl = '/home/dodinh/data/ngrams/{}_all_{}grams-errors.csv2'

google_ngram_path = ''
fname = 'googlebooks-eng-all-{}gram-20120701-{}.txt'


def extract_counts(ngrams, n, output_path, error_path):
    ngrams = [ngram.lower() for ngram in ngrams]
    ngram_dict = defaultdict(set)
    for ngram in ngrams:
        if ngram[0].isdigit():
            index = ngram[0]
        elif ngram[0] in string.punctuation:  #['_', '-', '.', ',']:
            index = 'punctuation'
        elif ngram[0].isalpha():
            tmp = ngram.replace('\'', '')  # single quotes are omitted in ngram data indexing
            tmp = tmp.replace('ä', 'a').replace('ö', 'o').replace('ü', 'u')  # umlauts are indexed by their non-umlaut variant
            index = re.sub('[^a-z]', '_', tmp[:2])
        else:
            index = 'other'
        ngram_dict[index].add(ngram)

    with open(output_path, 'w') as fw, open(error_path, 'w') as fe:
        for index, ngrams in ngram_dict.items():
            logging.info('Handling index [%s] with %s ngrams', index, len(ngrams))
            num_ngrams = len(ngrams)
            path = os.path.join(ngram_cache_path, fname).format(n, index)  # path to consolidated google ngram file
            with open(path, 'r') as fr:
                i = 1
                for line in fr:
                    line = line.strip().split('\t')
                    if line[0] in ngrams:
                        total = sum([int(year_counts.split(' ')[1]) for year_counts in line[1:]])
                        fw.write('{}\t{}\n'.format(line[0], total))
                        logging.info('Found ngram "%s" (%s) [%s/%s]', line[0], total, i, num_ngrams)
                        i += 1
                        ngrams.remove(line[0])
            if len(ngrams) > 0:
                logging.warn('Missed these ngrams in [%s]: %s', index, ', '.join(ngrams))
                fe.write('{}\t{}\n'.format(index, ','.join(ngrams)))
            print('')


def humour_ngrams(output_path):
    if os.path.exists(output_path):
        print('Output file [{}] exists, skipping.'.format(output_path))
        exit()

    ngrams = set()
    with open(humour_path, 'r') as fr:  #, open(output_path, 'w') as fw:
        reader = csv.reader(fr, delimiter='\t')
        next(reader)  # skip header row
        for row in reader:
            for ngram in nltk.ngrams(nltk.tokenize.word_tokenize(row[2]), n):
               ngrams.add(' '.join(ngram).lower())
            for ngram in nltk.ngrams(nltk.tokenize.word_tokenize(row[3]), n):
               ngrams.add(' '.join(ngram).lower())
    return ngrams


def metaphor_ngrams(output_path):
    if os.path.exists(output_path):
        print('Output file [{}] exists, skipping.'.format(output_path))
        exit()

    ngrams = set()
    vuamc = Vuamc(vuamc_path)
    for text in vuamc:
        for sentence in text.sentences:
             for i in range(0, len(sentence.tokens) - 1):
                  ngrams.add('{} {}'.format(sentence.tokens[i].covered_text, sentence.tokens[i+1].covered_text).lower())
    logging.info('%s %s-grams total', len(ngrams), n)
    return ngrams


if __name__ == '__main__':
    if not sys.argv or sys.argv[1] not in ['humour', 'metaphor']:
        logging.error('First argument should be "humour" or "metaphor"')
        exit()

    output_path = output_path_tmpl.format(task, n)
    error_path = error_path_tmpl.format(task, n)
    if sys.argv[1] == 'humour':
        ngrams = humour_ngrams(output_path)
    elif sys.argv[1] == 'metaphor':
        ngrams = metaphor_ngrams(output_path)
    extract_counts(ngrams, n, output_path, error_path)
