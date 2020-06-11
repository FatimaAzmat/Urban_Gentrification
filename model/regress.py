#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:20:26 2020

@author: manasip
"""

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

from userinput import rseed

def classifier_report(model, clf, y_true, y_hat):
    '''
    '''
    df_metric = pd.DataFrame(index=range(1),columns=['Model','R2-Score','MSE','MAE','Explained Variance'])
    
    df_metric['Model']= model
    df_metric['R2-Score'] = metrics.r2_score(y_true, y_hat)
    df_metric['MSE'] = metrics.mean_squared_error(y_true, y_hat)
    df_metric['MAE'] = metrics.mean_absolute_error(y_true, y_hat)
    df_metric['Explained Variance'] = metrics.explained_variance_score(y_true, y_hat)
    return df_metric


def linreg(model, df_xtrain, df_xtest, df_ytrain, df_ytest, rseed):
    '''
    '''
    
    #list of features
    preds_ls = list(df_xtrain.columns)  
    results_dict = dict()  

  
    for p in preds_ls:
        clf = linear_model.SGDRegressor(loss='squared_loss', penalty=None, random_state=rseed, max_iter=5000, tol=1e-3) 
        clf.fit(df_xtrain[[p]],df_ytrain)
        y_pred = clf.predict(df_xtest[[p]])
        sc  = metrics.r2_score(df_ytest, y_pred, multioutput='variance_weighted')
        mse = metrics.mean_squared_error(df_ytest, y_pred)  
        mae = metrics.mean_absolute_error(df_ytest, y_pred)  
        var = metrics.explained_variance_score(df_ytest, y_pred)
    
        results_dict[p] = [sc, mse, mae, var] 

    results = pd.DataFrame.from_dict(results_dict, orient='index').sort_values(by=0, ascending=False)
    results.reset_index(inplace=True)
    results.columns = ['Feature','R2-Score','MSE','MAE','Explained Variance']
    results.insert(loc=0, column="Model", value= model)
    
    return results


def multireg(model, df_xtrain, df_xtest, df_ytrain, df_ytest):
    '''
    '''
    
    clf = linear_model.LinearRegression(fit_intercept=True, copy_X=True)
    clf.fit(df_xtrain,df_ytrain)
    y_pred = clf.predict(df_xtest)
    
    #Dataframe coefficients
    df_coef = pd.DataFrame(list(df_xtrain.columns))
    df_coef.insert(len(df_coef.columns), "Coef Values", clf.coef_.transpose())
    df_coef.rename(columns={0:"Feature"}, inplace=True)
    df_coef.insert(0, "Model", model)
     
    return df_coef, classifier_report(model, clf, df_ytest, y_pred)

def randomforest(model, df_xtrain, df_xtest, df_ytrain, df_ytest, rseed):
    '''
    '''
    
    clf = ensemble.ExtraTreesRegressor(n_jobs=-1, random_state=rseed, n_estimators=100)  
    clf.fit(df_xtrain, df_ytrain)
    y_pred = clf.predict(df_xtest)
    
    #Dataframe features
    df_fi = pd.DataFrame.from_dict({'Feature': df_xtest.columns.values, \
                             'Importance': clf.feature_importances_})
    
    df_fi.sort_values(by='Importance', ascending=False, inplace=True)
    df_fi.insert(0, "Model", model)

    return df_fi, classifier_report(model, clf, df_ytest, y_pred)



def randomforest_tune(model, df_xtrain, df_xtest, df_ytrain, df_ytest, rseed):
    '''
    '''
    
    param_grid = {
    "n_estimators"      : [int(x) for x in np.arange(start=40, stop=251, step=20)] ,
    "max_depth"         : [None], 
    "min_samples_leaf"  : [1,2,4], 
    "max_features"      : [None]
    }
    
    clf = ensemble.ExtraTreesRegressor(n_jobs=-1, random_state=rseed) 
    
    start = timer()
    
    cv = model_selection.GridSearchCV(estimator=clf, param_grid=param_grid, cv=4, \
                                  n_jobs=-1, verbose=0, scoring='neg_mean_squared_error')
    cv.fit(df_xtrain, df_ytrain)
    
    duration = timer() - start
    
    #Compute best estimator
    best_clf = cv.best_estimator_ 
    best_clf.fit(df_xtrain, df_ytrain)
    y_pred  = best_clf.predict(df_xtest)

    
    #Dataframe features
    df_fi = pd.DataFrame.from_dict({'Feature': df_xtest.columns.values, \
                             'Importance':best_clf.feature_importances_}) 
    
    df_fi.sort_values(by='Importance', ascending=False, inplace=True)
    df_fi.insert(0, "Model", model)
    
    return df_fi, classifier_report(model, clf, df_ytest, y_pred)










    
    


    

