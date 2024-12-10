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


path = os.path.dirname(__file__)
sys.path.append(os.path.join(path, '../pstal-etu/lib/'))

with open('matriceE.bin', 'rb') as file:
    E = pickle.load(file)
with open('matriceT.bin', 'rb') as file:
    T = pickle.load(file)
with open('matriceP.bin', 'rb') as file:
    P = pickle.load(file)

# liste des étiquettes
t = list(set(itertools.chain(*[key.split(',') for key in T.keys() if ',' in key])))

def viterbi(seq, P, E, T, t):
    n = len(seq)
    N = len(t)
    delta = np.full((N, n), fill_value=Util.PSEUDO_INF)
    phi = np.zeros((N, n), dtype=int)

    for j in range(N):
        delta[j, 0] = P.get(t[j], Util.PSEUDO_INF) + E.get(f"{t[j]},{seq[0]}", E['<<<OOV>>>'].get(t[j], Util.PSEUDO_INF))

    for k in range(1, n):
        for j in range(N):
            temp = []
            for i in range(N):
                trans_prob = T.get(f"{t[i]},{t[j]}", T['<<<OOV>>>'].get(t[i], Util.PSEUDO_INF))
                temp.append(delta[i, k - 1] + trans_prob)
            delta[j, k] = min(temp) + E.get(f"{t[j]},{seq[k]}", E['<<<OOV>>>'].get(t[j], Util.PSEUDO_INF))
            phi[j, k] = np.argmin(temp)

    pred_index = [np.argmin(delta[:, n - 1])]
    for k in range(n - 2, -1, -1):
        pred_index.insert(0, phi[pred_index[0], k + 1])
    return [t[i] for i in pred_index]


file_test = os.path.join(path, '../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.test')
# with open(file_test, 'r') as file:
#     reader = CoNLLUReader(file)
#     for sent in reader.readConllu():
#         seq = [str(token) for token in sent]
#         pred = viterbi(seq, P, E, T, t)
#         print(f"Phrase : {sent.metadata['text']}")
#         print(f"Cibles : {CoNLLUReader.to_bio(sent)}")
#         print(f"Prédictions : {pred}")

with open(file_test, 'r') as file:
    reader = CoNLLUReader(file)
    with open('prediction_file.conllu', 'w') as pred_file:
        for sent in reader.readConllu():
            seq = [str(token) for token in sent]
            pred = viterbi(seq, P, E, T, t)
            sent_pred = CoNLLUReader.from_bio(pred)
            pred_file.write(str(sent_pred) + '\n')