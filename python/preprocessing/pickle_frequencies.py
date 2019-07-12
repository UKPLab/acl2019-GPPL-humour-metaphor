#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle

frequencies = dict()
with open('freq_tokens.csv', 'r') as f:
    for line_no, line in enumerate(f):
        token, count = line.split('\t')
        count = int(count)
        if count >= 10000:  # only save entries with over 10k occurrences
            freqs[token] = count

with open('wikipedia_2017_frequencies.pkl', 'wb') as f:
    pickle.dump(frequencies, f)

print('Original frequencies file has %s lines.', lines)
print('Pickled frequencies dictionary has %s entries (%s%).', len(freqs), len(freqs)/line_no)
