import torch
import sys 
import os
path = os.path.dirname(__file__)
sys.path.append(os.path.join(path,'../pstal-etu/lib/'))
from conllulib import Util, CoNLLUReader
from collections import defaultdict, Counter

file_test = os.path.join(path, '../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.small')

dict_w_t = defaultdict(int)
dict_t_t = defaultdict(int)
dict_t = defaultdict(int)
dict_s_t = defaultdict(int)
S = 0

sentence = []
encoded = []
with open(file_test,'r') as file :
        reader = CoNLLUReader(file)
        
        for sent in reader.readConllu():
            
            # print(sent.metadata['text'])
            #print()
            l = CoNLLUReader.to_bio(sent)
            #print(str(sent[0]))
            dict_s_t[str(sent[0])] += 1
            S += 1
            for i,token in enumerate(sent):
                  dict_t[l[i]] += 1
                  dict_w_t[str(token)+','+l[i]] +=1
                  try:
                        dict_t_t[l[i]+','+l[i+1]] += 1
                  except:
                        pass
print(S)
E = defaultdict()
T = defaultdict()
p = defaultdict()

for key in dict_w_t.keys():
      w,t = key.split(',')[0], key.split(',')[1]
      #print(w,t)
      #print(dict_t[t], t)
      if dict_t[t] != 0 :
            E[t + ',' + w] = dict_w_t[key]/dict_t[t]

for key in dict_t_t.keys():
      ti = key.split(',')[0]
      if dict_t[ti]!=0:
            T[key] = dict_t_t[key]/dict_t[ti]

for key in dict_s_t.keys():
      p[key] = dict_s_t[key]/S

