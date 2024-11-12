import torch
import sys 
import os
path = os.path.dirname(__file__)
sys.path.append(os.path.join(path,'../pstal-etu/lib/'))
from conllulib import Util, CoNLLUReader
from conllu import parse_incr, parse

# on doit avoir une accuracy de 80/90%
# Attention à utiliser le même vocabulaire pour le train et le dev 
file_test = os.path.join(path, '../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.small')





################ UTILS 
def pad_tensor(X, max_len): 
    res = torch.full((len(X), max_len), 0) # padding 
    for (i, row) in enumerate(X) : 
        x_len = min(max_len, len(X[i])) 
        res[i,:x_len] = torch.LongTensor(X[i][:x_len]) 
    return res



################



max_len=1000

with open(file_test,'r') as file :
    reader = CoNLLUReader(file)
    col_name_dict = {"form":["<PAD>", "<UNK>"], "upos":["<PAD>"]}
    int_list, vocab = reader.to_int_and_vocab(col_name_dict)

print(int_list) # renvoi un dict avec la key 'form' et 'upos'
print( vocab) # map mot vers encodage 


sentence = pad_tensor(int_list['form'], max_len)
description = pad_tensor(int_list['upos'],max_len)

dataloader = Util.dataloader(sentence, description)