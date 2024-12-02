import torch
import sys 
import os
path = os.path.dirname(__file__)
sys.path.append(os.path.join(path,'../pstal-etu/lib/'))
from conllulib import Util, CoNLLUReader
from conllu import parse_incr, parse

file_test = os.path.join(path, '../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.small')

from collections import defaultdict, Counter
import pickle

# Initialisation dictionnaires de comptage
count_word_tag = defaultdict(Counter)
count_tag_transition = defaultdict(Counter)
count_tag = Counter()
count_initial_tag = Counter()


corpus = list(CoNLLUReader.readConllu(file_test))
bio_corpus = [CoNLLUReader.to_bio(sentence) for sentence in corpus]

for sentence in bio_corpus:
    prev_tag = None
    for i, token in enumerate(sentence):
        word = token['form']
        tag = token['bio']
        
        count_word_tag[word][tag] += 1
        count_tag[tag] += 1
        
        if prev_tag is not None:
            count_tag_transition[prev_tag][tag] += 1
        else:
            count_initial_tag[tag] += 1
        
        prev_tag = tag

emission = {}
for tag, word_counts in count_word_tag.items():
    total_tag = count_tag[tag]
    emission[tag] = {word: log_cap(total_tag) - log_cap(count) 
                     for word, count in word_counts.items()}

with open('hmm_params.pkl', 'wb') as f:
    pickle.dump({'emission': emission,
                 'transition': count_tag_transition,
                 'initial': count_initial_tag}, f)
