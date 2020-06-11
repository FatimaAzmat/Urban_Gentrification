#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:12:39 2020

@author: manasip
"""

import os
import pandas as pd
from functools import reduce
from sklearn.model_selection import train_test_split

from userinputs import wd, cfilename, MODELS, tsize, rseed, model1, model2, model3, dropcol_features, dropcol_input
import process as pp
import regress as rg


##---------------------------------------------------------------##
##              Create df to store results
##---------------------------------------------------------------##


#Linear Regression
reslinear = pd.DataFrame(columns=['Model', 'Feature', 'R2-Score', 'MSE', 'MAE', 'Explained Variance'])

#Multi Regression
resmulticoef = pd.DataFrame(columns=['Model', 'Feature', 'Coef Values'])
resmulti = pd.DataFrame(columns=['Model', 'R2-Score', 'MSE', 'MAE', 'Explained Variance'])

#Random Forest without Hyperparameter tuning
resrfcoef = pd.DataFrame(columns=['Model', 'Feature', 'Importance'])
resrf = pd.DataFrame(columns=['Model', 'R2-Score', 'MSE', 'MAE', 'Explained Variance'])

#Random Forest without Hyperparameter tuning
resrfcoef_tune = pd.DataFrame(columns=['Model', 'Feature', 'Importance'])
resrf_tune = pd.DataFrame(columns=['Model', 'R2-Score', 'MSE', 'MAE', 'Explained Variance'])


##---------------------------------------------------------------##
##              Import files
##---------------------------------------------------------------##

#Features
for model in MODELS:
    
    
    df_features= []
    for yr in model["numyr"]:
        df_features.append(pd.read_csv(os.path.join(wd, '20'+ str(yr) + cfilename), dtype={"geo11":str}))

    #SES Inputs
    df_input = pd.read_csv(os.path.join(wd, model["ses_input"]), dtype={"geo11":str})

    df_features.append(df_input)

    #Target
    df_ses = pd.read_csv(os.path.join(wd, model["ses"]), dtype={"geo11":str})


##---------------------------------------------------------------##
##              Process files
##---------------------------------------------------------------##

    #Features
    for df in df_features:
        pp.chg_col(df)
        df.fillna(df.median(), inplace=True)
    
    #Target
    pp.chg_col(df_ses)
    
    #Drop columns from features - user input - TO WRITE THIS

    #if xvar!="all":
     #   for cols in df_features[0]:
      #      if 
        
    #Drop columns from features
    for df in df_features[:-1]:
        df.drop(dropcol_features, axis=1, inplace=True)
    
    #Drop columns from input to ses
    df_features[-1].drop(dropcol_input, axis=1, inplace=True)

    
    #Rename columns
    i=0
    for yr in model["numyr"]:
        df_features[i].columns = [str(col) + '_' + str(yr) for col in df_features[i].columns]
        i += 1


    #Merge all features 
    df_xvar = reduce(lambda left,right: pd.merge(left,right,on='geo11'), df_features)


##---------------------------------------------------------------##
##              Train-Test Datasets
##---------------------------------------------------------------##

    #Check if all the indexes match
    df_xvar.sort_index(inplace=True)
    df_ses.sort_index(inplace=True)
    
    
    #Drop rows
    droprow = list(set(df_xvar.index.values) - set(df_ses.index.values))
    if droprow != []:
        df_xvar.drop(droprow, axis=0, inplace=True)
    else:
        droprow = list(set(df_ses.index.values) - set(df_xvar.index.values))
        df_ses.drop(droprow, axis=0, inplace=True)
        
    df_xvar.sort_index(inplace=True)
    df_ses.sort_index(inplace=True)
        

    #Train-test split
    df_xtrain, df_xtest, df_ytrain, df_ytest = train_test_split(df_xvar, df_ses, test_size=tsize, random_state=rseed)


    #Get column names of columns to normalize
    normcol = list(df_input.columns)
    for cols in df_xvar.columns:
        if ("age_" in cols.lower()) or ("local morans" in cols.lower()) or ("value spatial" in cols.lower()):
            normcol.append(cols)
            
    #Normalize columns
    df_xtrain, df_xtest = pp.process_normalize(df_xtrain, df_xtest, normcol)


##---------------------------------------------------------------##
##              Regressions
##---------------------------------------------------------------##
    
    #Linear Regression
    df_linear = rg.linreg(model["name"], df_xtrain, df_xtest, df_ytrain["scores_pr_asc"], df_ytest["scores_pr_asc"], rseed)
    reslinear = reslinear.append(df_linear)


    #Mulit Regression
    df_multicoef, df_multires = rg.multireg(model["name"], df_xtrain, df_xtest, df_ytrain["scores_pr_asc"], df_ytest["scores_pr_asc"])
    
    resmulticoef = resmulticoef.append(df_multicoef)
    resmulti = resmulti.append(df_multires)

    #Random Forest without Hyperparameter tuning
    df_rfcoef, df_rf = rg.randomforest(model["name"], df_xtrain, df_xtest, df_ytrain["scores_pr_asc"], df_ytest["scores_pr_asc"], rseed)
    
    resrfcoef = resrfcoef.append(df_rfcoef)
    resrf = resrf.append(df_rf)
    
    #Random Forest with Hyperparameter tuning
    df_rfcoef_tune, df_rf_tune = rg.randomforest_tune(model["name"], df_xtrain, df_xtest, df_ytrain["scores_pr_asc"], df_ytest["scores_pr_asc"], rseed)
    
    resrfcoef_tune = resrfcoef_tune.append(df_rfcoef_tune)
    resrf_tune = resrf_tune.append(df_rf_tune)
    

reslinear.to_csv("linear_regression_features.csv")

resmulticoef.to_csv("multi_regression_features.csv")
resmulti.to_csv("multi_regression_metrics.csv")

resrfcoef.to_csv("rf_wo_features.csv")
resrf.to_csv("rf_wo_metrics.csv")

resrfcoef_tune.to_csv("rf_tune_features.csv")
resrf_tune.to_csv("rf_tune_metrics.csv")
    
    
    
    
    
    
    
    