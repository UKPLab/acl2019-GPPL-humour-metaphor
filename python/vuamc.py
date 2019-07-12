#!/usr/bin/env python
# -*- coding: utf8 -*-

import logging
import os
import pickle
import re
import sys
from collections import namedtuple
from lxml import etree

logging.basicConfig(level=logging.INFO)

Text = namedtuple("Text", "id sentences")
Sentence = namedtuple("Sentence", "number tokens")
Token = namedtuple("Token", "covered_text original_text lemma pos metaphor score")

class Sentence(Sentence):
    def pp(self, pos=False):
        if pos:
            return ''.join([t.pos + '#' + t.original_text for t in self.tokens])
        else:
            return ''.join([t.original_text for t in self.tokens])


ns = {'TEI': 'http://www.tei-c.org/ns/1.0'}
pkl_path = 'vuamc.pkl'
default_path = '/home/dodinh/data/VU Amsterdam Metaphor Corpus/2541/VUAMC.xml'


class Vuamc:

    def __init__(self, path=default_path):
        if os.path.isfile(pkl_path):
            self.texts = self.unpickle_vuamc()
        else:
            self.texts = self.extract_vuamc(path)
            self.pickle_vuamc()
        self.len = len(self.texts)

    def __len__(self):
        return len(self.texts)

    def __iter__(self):
        return self.texts.__iter__()

    def __next__(self):
        return self.texts.__next__()

    def print_neat(self):
        print("Text: id=%s" % self.text.id)
        for s in self.text.sentences:
            print("  Sentence: n=%s" % s.number)
            for t in s.tokens:
                print("    Token %s: lemma=%s, pos=%s, met=%s" % t)

    def get(self, vuamc_id):
        text_id, sentence_id, token_id = (int(i) for i in vuamc_id.split('.'))
        text = self.texts[text_id]
        sentence = text.sentences[sentence_id]
        token = sentence.tokens[token_id]
        return text, sentence, token

    def get_text(self, vuamc_id):
        return self.get(vuamc_id)[0]

    def get_sentence(self, vuamc_id):
        return self.get(vuamc_id)[1]

    def get_token(self, vuamc_id):
        return self.get(vuamc_id)[2]

    @staticmethod
    def extract_vuamc(path):
        logging.info('Reading VUAMC from xml')
        root = etree.parse(path).getroot()
        texts = []
        for xml_text in root.xpath('./TEI:text/TEI:group/TEI:text', namespaces=ns):
            sentences = []
            for xml_sentence in xml_text.xpath('.//TEI:s', namespaces=ns):
                tokens = []
                xml_tokens = xml_sentence.xpath('.//TEI:w | .//TEI:c', namespaces=ns)
                for i, xml_token in enumerate(xml_tokens):
                    is_met = None
                    score = None
                    # original text from the XML; newlines removed, spaces condensed
                    original_text = re.sub(r'\s+$', ' ', xml_token.text) if xml_token.text is not None else ''
                    if i+1 < len(xml_tokens) and xml_tokens[i+1].attrib['type'] in ['POS', 'PUC', 'PUN']:
                         original_text = original_text.rstrip()
#                    xml_seg = xml_token.find('./TEI:seg', ns)
#                    if xml_seg is not None and len(xml_seg) > 0:
#                        if len(xml_seg) > 1:
#                            for x in xml_seg:
#                                print(x.text, x.attrib)
#                            input("wowie")
#                        else:
#                            print(xml_seg)
#                            xml_seg = xml_seg[0]
#                        is_met = (xml_seg.attrib['function'] == 'mrw' and xml_seg.attrib['type'] == 'met')
#                        score = xml_seg.attrib['score'] if 'score' in xml_seg.attrib else None
#                        original_text = re.sub(r'\s+$', ' ', (xml_seg.text if xml_seg.text is not None else '') + original_text)

                    # todo cycle through xpath('.//TEI:seg | text')

                    xml_segs = xml_token.xpath('.//TEI:seg', namespaces=ns)
                    if len(xml_segs) > 0:
                        lemma = xml_token.attrib['lemma'] if 'lemma' in xml_token.attrib else xml_token.covered_text
                        pos = xml_token.attrib['type']
                        for xml_seg in xml_segs:
                            is_met = (xml_seg.attrib['function'] == 'mrw' and xml_seg.attrib['type'] == 'met')
                            score = xml_seg.attrib['score'] if 'score' in xml_seg.attrib else None
                            original_text2 = re.sub(r'\s+$', ' ', (xml_seg.text if xml_seg.text is not None else '') + original_text)
                            # covered text (whitespace removed), lemma, POS tag
                            covered_text = original_text2.strip()
                            # create token
                            token = Token(covered_text, original_text2, lemma, pos, is_met, score)
                            tokens.append(token)
                    else:
                        # covered text (whitespace removed), lemma, POS tag
                        covered_text = original_text.strip()
                        lemma = xml_token.attrib['lemma'] if 'lemma' in xml_token.attrib else covered_text
                        pos = xml_token.attrib['type']
#                        if covered_text == 'taffeta':
#                            print('taffeta')
#                            print(xml_seg.attrib)
#                            input("STOPPO")
                        # create token
                        token = Token(covered_text, original_text, lemma, pos, is_met, score)
                        tokens.append(token)
                # combine tokens to sentence
                sentence = Sentence(xml_sentence.attrib['n'], tokens)
                sentences.append(sentence)
            # combine sentences to text
            text = Text(xml_text.attrib['{http://www.w3.org/XML/1998/namespace}id'], sentences)
            texts.append(text)
        return texts

    def pickle_vuamc(self):
        logging.info('Saving VUAMC to pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.texts, f)

    @staticmethod
    def unpickle_vuamc():
        logging.info('Loading VUAMC from pkl')
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)


def embs():
    path='/home/dodinh/data/embeddings/word2vec/GoogleNews-vectors-negative300.vocab_sub_VUAMC'
    logging.info('Loading embeddings')
    embeddings = dict()
    with open(path, 'r') as f:
        for line in f:
            line = line.split(' ')
            embeddings[line[0]] = [float(a) for a in line[1:]]
    return embeddings


if __name__ == '__main__':
#    vuamc = Vuamc(default_path)
    vuamc = Vuamc(default_path.replace('VUAMC', 'VUAMC_with_novelty_scores'))
    print()
    print("no texts:", len(vuamc))
    print("no sentences:", sum(len(text.sentences) for text in vuamc))
    print("no tokens:", sum(len(s.tokens) for text in vuamc for s in text.sentences))

    if len(sys.argv) > 1:
        if sys.argv[1] == 'embs':
            m = embs()
            cov = 0
            unc = 0
            for te in vuamc:
                for se in te.sentences:
                    for to in se.tokens:
                        if to.covered_text in m:
                            cov += 1
                        else:
                            unc += 1
            print('covered', cov, (cov+unc), cov/(cov+unc))
            print('uncover', unc, (cov+unc), unc/(cov+unc))
            exit()

        print()
        for vuamc_id in sys.argv[1:]:
            if re.match('^\d+\.\d+\.\d+$', vuamc_id) is not None:
                text, sentence, token = vuamc.get(vuamc_id)
                print(token)
                print(sentence.pp())
#                for s in text.sentences[:10]:
#                    print(s.pp())
