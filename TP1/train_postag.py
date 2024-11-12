import torch
import sys 
import os
path = os.path.dirname(__file__)
sys.path.append(os.path.join(path,'../pstal-etu/lib/'))
from conllulib import Util, CoNLLUReader
from conllu import parse_incr, parse


file_test = os.path.join(path, '../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.small')


# def load_from_file(path):
#     """
#     return a list of tuple with 
#     """
# for sent in parse_incr(open(file_test, encoding='UTF-8')):
#     print(" ".join(tok["upos"] for tok in sent))
#     print()
#     print(" ".join(tok["form"] for tok in sent))

with open(file_test,'r') as file :
    reader = CoNLLUReader(file)
    col_name_dict = {"form":["<PAD>", "<UNK>"], "upos":["<PAD>"]}
    int_list, vocab = reader.to_int_and_vocab(col_name_dict)

print(int_list)

print( vocab)

# sentence = parse(open(file_test, encoding='UTF-8').readline())
# print(sentence)