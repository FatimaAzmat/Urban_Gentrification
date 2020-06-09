#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:28:40 2020

@author: manasip
"""
import os
import pandas pd
import process as pp
from userinput import dyr, ffolder, ffilename, sfolder, sfilename, wd

for yr in dyr:
    print(yr)
    df_features = pd.read_csv(os.path.join(ffolder, str(yr)+ffilename))
    pp.chg_col(df_features)
    df_features = pp.convert_acs_encoding(df_features)
    df_features.drop(pp.find_unpopulated_tracts(df_features), inplace=True)
    pp.impute_missing_values(df_features)
    df_features.to_csv(str(yr) +"_features_clean.csv")
    

for yr in dyr:
    print(yr)
    df_ses = pd.read_csv(os.path.join(sfolder, str(yr)+sfilename))
    pp.chg_col(df_ses)
    df_ses = pp.convert_acs_encoding(df_ses)
    df_ses.drop(pp.find_unpopulated_tracts(df_ses), inplace=True)
    pp.impute_missing_values(df_ses)
    df_ses.to_csv(str(yr) +"_ses_clean.csv")    
    
    
    

