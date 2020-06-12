#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:28:40 2020

@author: manasip
"""
import os
import pandas as pd
import process as pp
from userinput import wd, numyr, ffolder, ffilename, cfilename



for yr in numyr:
    df_features = pd.read_csv(os.path.join(ffolder, str(yr)+ ffilename), dtype={'geo11':str})
    pp.chg_col(df_features)
    df_features = pp.convert_acs_encoding(df_features)
    df_features.drop(pp.find_unpopulated_tracts(df_features), inplace=True)
    pp.impute_missing_values(df_features)
    df_features.to_csv(str(yr) +cfilename)
    
