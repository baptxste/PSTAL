from conllu import parse_incr
import sys
import os
path = os.path.dirname(__file__)
sys.path.append(os.path.join(path,'../pstal-etu/sequoia'))

for sent in parse_incr(open("pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.small", encoding='UTF-8')):
  print(" ".join(tok["upos"] for tok in sent))


def data(file_name):
  

