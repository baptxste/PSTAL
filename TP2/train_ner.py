import torch
import sys 
import os
path = os.path.dirname(__file__)
sys.path.append(os.path.join(path,'../pstal-etu/lib/'))
from conllulib import Util, CoNLLUReader
from collections import defaultdict, Counter

file_test = os.path.join(path, '../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.small')


sentence = []
encoded = []
with open(file_test,'r') as file :
        reader = CoNLLUReader(file)
        
        for sent in reader.readConllu():
            for token in sent : 
                  print(token)
            # print(sent.metadata['text'])
            print( CoNLLUReader.to_bio(sent))

