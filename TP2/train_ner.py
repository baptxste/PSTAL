import torch
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

file_test = os.path.join(path, '../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.train')

# Initialisation des dictionnaires
dict_w_t = defaultdict(int)
dict_t_t = defaultdict(int)
dict_t = defaultdict(int)
dict_s_t = defaultdict(int)
S = 0
alpha = 0.1  # Paramètre de lissage

with open(file_test, 'r') as file:
    reader = CoNLLUReader(file)
    for sent in reader.readConllu():
        l = CoNLLUReader.to_bio(sent)
        dict_s_t[l[0]] += 1
        S += 1
        for i, token in enumerate(sent):
            dict_t[l[i]] += 1
            dict_w_t[f"{str(token)},{l[i]}"] += 1
            if i < len(sent) - 1:
                dict_t_t[f"{l[i]},{l[i + 1]}"] += 1

# Calcul des probabilités lissées
Vw = len(dict_w_t.keys())
Vt = len(dict_t.keys())
E = {}
for key in dict_w_t.keys():
    try:
      w, t = key.split(',')
    except : 
        w=","
        t = key.split(',')[-1]
    E[f"{t},{w}"] = math.log(dict_t[t] + Vw * alpha) - math.log(dict_w_t[key] + alpha)
E['<<<OOV>>>'] = {t: math.log(dict_t[t] + Vw * alpha) - math.log(alpha) for t in dict_t.keys()}

T = {}
for key in dict_t_t.keys():
    ti, tj = key.split(',')
    T[key] = math.log(dict_t[ti] + Vt * alpha) - math.log(dict_t_t[key] + alpha)
T['<<<OOV>>>'] = {ti: math.log(dict_t[ti] + Vt * alpha) - math.log(alpha) for ti in dict_t.keys()}

p = {}
for key in dict_s_t.keys():
    p[key] = math.log(S + Vt * alpha) - math.log(dict_s_t[key] + alpha)

# Sauvegarde des paramètres
with open('matriceE.bin', 'wb') as f:
    pickle.dump(E, f)
with open('matriceT.bin', 'wb') as f:
    pickle.dump(T, f)
with open('matriceP.bin', 'wb') as f:
    pickle.dump(p, f)