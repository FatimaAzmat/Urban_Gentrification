#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 00:28:30 2020

@author: manasip
"""
%reload_ext autoreload

%autoreload 2
# For reproducibility
#import random
import numpy as np
#r_state = 42
#random.seed(r_state) 
#np.random.seed(r_state)

import os
import re
import pandas as pd
import seaborn as sns

import sklearn

from sklearn.preprocessing import scale
from sklearn import linear_model
from sklearn import tree
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics  
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#from sklearn.externals.six import StringIO
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import f_regression

from timeit import default_timer as timer
import datetime
from functools import reduce

#Import global variables
from input_variables import wd, SF, SF_DICT, STATE_SF, numyr, tsize, rseed, outcome, dropcol

from input_variables import dropcol

import input_variables as iv

#Import preprocessing module
import preprocessing as pp


#Import target files
df_ses = pd.read_csv(outcome)

#Input files
df_features = []
for yr in numyr:
    folder = os.path.join(wd, 'features_dataset')
    filename = yr + "_acs5_features.csv"
    print(filename)
    df_features.append(pd.read_csv(os.path.join(folder, filename)))

#Set geo11 as index for target and features   
pp.chg_col(df_ses, "geo11")

for df in df_features:
    pp.chg_col(df, "geo11")
    
#Drop tracts from features that are not in target - can make tracts be adapted?
for df in df_features:
    droprow = list(set(df.index.values) - set(df_ses.index.values))
    df.drop(droprow, axis=0, inplace=True)

#Convert Invalid values to nans
for df in df_features:
    df = pp.convert_acs_encoding(df)


#Impute missing values
for state, sf in STATE_SF.items():
    for df in df_features:
        pp.fillna(df, state, gmean=True, shapefile=sf)
    
#Drop columns
for df in df_features:
    df.drop(dropcol, axis=1, inplace=True)
    
#Rename columns
i=0
for yr in numyr:
    df_features[i].columns = [str(col) + '_' + yr for col in df_features[i].columns]


#Merge dataframes
df_xvar = reduce(lambda left,right: pd.merge(left,right,on='geo11'), df_features)

#Reference
#https://stackoverflow.com/questions/23668427/pandas-three-way-joining-multiple-dataframes-on-columns

nans_b = nans



    

    

    
    

