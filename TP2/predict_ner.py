import torch
import sys 
import os
path = os.path.dirname(__file__)
sys.path.append(os.path.join(path,'../pstal-etu/lib/'))
from conllulib import Util, CoNLLUReader
from collections import defaultdict, Counter
import pickle
import numpy as np
import itertools

with open('matriceE.bin','rb')as file : 
    E = pickle.load(file)
with open('matriceT.bin','rb')as file : 
    T = pickle.load(file)
with open('matriceP.bin','rb')as file : 
    P = pickle.load(file)
# print(E)
# liste des étiquettes uniques
t = list(set(itertools.chain(*[key.split(',') for key in T.keys()]))) # * permet d'éclater une liste
def viterbi(seq:list, P:dict=P, E:dict=E, T:dict=T,t:list=t):
    """
    seq : liste des mots de la phrase
    """
    n = len(seq)
    # print(f" n = {n}")
    N = len(t)
    # print(f" N = {N}")
    delta = np.full((N,n),fill_value=Util.PSEUDO_INF)
    phi = np.full((N,n),fill_value=0.0)
    for j in range(N):
        try : # gestion des oov à faire
            delta[j,1] = P[t[j]] + E[str(t[j]+','+str(seq[0]))]
        except : pass 
    for j in range(N):
        for k in range(2,n):
            try : 
                temp = [delta[i, k-1]+T[str(t[i]+','+t[j])] for i in range(N)]
                mini = min(temp)
                delta[j,k] = mini + E[str(t[j]+','+str(seq[k]))]
                phi[j,k] = temp.index(mini)
            except : pass

    # on prédit les étiquettes
    pred_index_inv = []
    pred_inv = []
    # print( delta.shape)

    l = [delta[i,n-1] for i in range(N)]
    pred_index_inv.append(l.index(min(l)))
    pred_inv.append(delta[l.index(min(l)),n-1])

    k = n-2 # en réalité c'es n-1 mais les indinces commencent a zero
    while k >= 0 :
        print(pred_index_inv[-1])
        pred_inv.append(phi[pred_index_inv[-1],k+1])
        pred_index_inv.append(t.index(pred_inv[-1]))
    
    return [t[i] for i in pred_index_inv[:: -1]]



file_test = os.path.join(path, '../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.small')

with open(file_test,'r') as file :
        reader = CoNLLUReader(file)
        
        for sent in reader.readConllu():
            seq = [str(token) for token in sent]
            # print(sent.metadata['text'])
            #print()
            l = CoNLLUReader.to_bio(sent)
            # print(l)
            # for token in sent:
            #     print(str(token))
            # print(seq)
            pred = viterbi(seq)
            print(f" SENT  : {sent.metadata['text']}")
            print(f" target : {l}")
            print(f" Pred : {pred}")