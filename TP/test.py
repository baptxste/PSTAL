import torch
import sys 
import os
path = os.path.dirname(__file__)
sys.path.append(os.path.join(path,'../pstal-etu/lib/'))
from conllulib import Util, CoNLLUReader
from conllu import parse_incr, parse
from collections import defaultdict, Counter
import pickle

file_test = os.path.join(path, '../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.small')


for sent in parse_incr(open(file_test, encoding='UTF-8')):
  #print(" ".join(tok["upos"] for tok in sent))
  print(sent)
