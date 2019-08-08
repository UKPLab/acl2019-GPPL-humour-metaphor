#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import nltk

input = 'results.tsv'
output = 'humour_all_words.txt'

tokens = set()
with open(input, 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for line_no, row in enumerate(reader):
      if line_no % 1000 == 0:
          print('Handling line {}'.format(line_no))
      try:
        for t in nltk.tokenize.word_tokenize(row[2]):
            tokens.add(t.lower())
        for t in nltk.tokenize.word_tokenize(row[3]):
            tokens.add(t.lower())
      except IndexError as e:
        print('line:', line_no, row)
        raise e

with open(output, 'w') as f:
    for t in tokens:
        f.write('{}\n'.format(t))
