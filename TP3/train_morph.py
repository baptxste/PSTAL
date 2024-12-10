# import torch
import sys 
import os
path = os.path.dirname(__file__)
sys.path.append(os.path.join(path,'../pstal-etu/lib/'))
from conllulib import Util, CoNLLUReader
from collections import defaultdict, Counter
import pickle
import math
import itertools


path = os.path.dirname(__file__)
sys.path.append(os.path.join(path, '../pstal-etu/lib/'))

# file_test = os.path.join(path, '../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.small')
file_test = os.path.join(path, '../pstal-etu/sequoia/tiny.conllu')


def load_data(file=file_test):
    data = []
    map_char = {"<pad>":0, "<unk>":1, "<esp>":2}
    with open(file_test, 'r') as file:
        reader = CoNLLUReader(file)
        feats = reader.morph_feats()
        print(feats)
        for sent in reader.readConllu():
            for tok in sent:
                print(tok, tok['feats'])
            text = " ".join([str(e) for e in sent])
            chars = ["<pad>"]
            in_enc = [0]
            ends = []
            for i,char in enumerate(text) : 
                if char ==" ":
                    char = "<esp>"
                    ends.append(i)
                if char not in map_char.keys():
                    map_char[char] = len(map_char.keys())
                in_enc.append(map_char[char])
                chars.append(char)
            ends.append(i)
            data.append([text ,chars, in_enc,ends])
            # bien lire la partie sur le modèle car tout dépend de ca 


load_data()