# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 19:04:46 2018

@author: luigimissri
"""

import pandas as pd

dat=pd.read_csv("C:/Users/luigimissri/Desktop/FILE DA SALVARE/Data Science/Knoledge and Data Mining/Progetto/dati_anal.csv")
del dat["Unnamed: 0"]
#identificativo=dat["id"]
#del dat["id"]


import numpy as np
from itertools import combinations

dati=np.array(dat)

alpha = 0.40
beta = 0.87
observation,variables = dati.shape
Omega = range(variables)
transactions = range(observation)

Fi=np.sum(dati,axis=0)


items=list(dat.columns.values)


def get_frequent_itemsets(dati,alpha):
    print(dati)
    L = {0:{tuple():observation}}
    Fi = np.sum(dati,axis=0)
    L[1] = {(i,):Fi[i] for i in Omega if Fi[i] >= alpha*observation}
    k = 1
    while L[k]:
        L[k+1] = {}
        for s1 in L[k]:
            for s2 in L[k]:
                s3 = set(s1)-set(s2)
                if len(s3) == 1:
                    s12 = set(s1) | set(s2)
                    s12_is_good = True
                    for i in s12:
                        if tuple(sorted(s12-{i})) not in L[k]:
                            s12_is_good = False
                            break
                    if s12_is_good:
                        s12 = tuple(sorted(s12))
                        Fs12 = np.sum(np.all(dati[:,s12],axis=1))
                        if Fs12 >= alpha*observation:
                            L[k+1][s12]=Fs12
        k += 1
    return L

def get_rules(L, beta):
    R = []
    for k in L:
        for s in L[k]:
            for j in range(1,len(s)):
                for sub_s in combinations(set(s),j):
                    if L[len(sub_s)][tuple(sorted(sub_s))]*beta <= L[k][s]:
                        R.append([sub_s,tuple(set(s)-set(sub_s))])
    return R

L = get_frequent_itemsets(dati,alpha)

for k in L:
    for S in L[k]:
        print([items[i] for i in S])





R = get_rules(L,beta)
if R:
    print("association rules")
    for r in R:
        print([items[i] for i in r[0]]," -> ",[items[i] for i in r[1]])
else:
    print("no association rules")
    
    