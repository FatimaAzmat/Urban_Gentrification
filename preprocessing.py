#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 00:02:49 2020

@author: manasip
"""

##Module for preprocessing data
"""
Usage
    
Import preprocessing as pp
    
df_acs = pp.convert_acs_encoding(df_acs)
"""


import pandas as pd
import geopandas as gpd
import os
import numpy as np
import pysal
from scipy.stats import gmean
from input_variables import SF_DICT
from sklearn.preprocessing import StandardScaler


def chg_col(df, colname):
    """
    Function sorts by column name, changes column type to str and sets column as index
    :params df(pandas dataframe): ACS data
            colname(str): Column to be formatted
    """
    df.sort_values(by=colname, inplace=True)
    df[colname] = df[colname].astype(str)
    df.set_index(colname, inplace=True)



def convert_acs_encoding(df):
    """
    Undefined/missing values in ACS dataset are assigned value -666666666.0
    Function replaces value with nan
    :param df(pandas dataframe): ACS data
    :data(pandas dataframe): ACS data, nans assigned to missing values
    """
    
    data = df.copy()
    nan_code = -666666666.0

    for col in data.columns:
        indexes = data[data[col] == nan_code].index
        data.loc[indexes, col] = np.nan

    return data


def find_unpopulated_tracts(df):
    """
    Drop unpopulated tracts assuming that all rows missing all data are
    unpopulated.
    :param df(pandas dataframe): ACS data
    :return:tracts
    """

    data = df.copy()
    narrow = data.loc[:, data.columns != 'state']
    tracts = narrow[narrow.isnull().all(axis=1)].index.values
    #print(type(tracts)) #Change this 

    return tracts



def findna(df):
    """
    Find columns with missing values and count the total number of missing
    values.
    :param df(pandas dataframe): ACS data
    :return: dictionary mapping column names to total counts of missing values
    """

    data = df.copy()

    cnt = lambda x: data[x].isna().sum()

    return {col: cnt(col) for col in data.columns if cnt(col)}


def merge_geodata(acs_data, state, shapefile):
    """
    Merge ACS data with a state's shapefile using the census tract id
    :param acs_data: DataFrame
    :param state: (str) state to use for shapefile
    :param shapefile: (str) file path to shapefile
    :return: GeoDataFrame
    """

    # Read the shapefile into a GeoDataFrame
    gdf = gpd.read_file(os.path.join(SF_DICT[state], shapefile))

    gdf.rename(columns={'GEOID10': 'geo11'}, inplace=True)

    gdf.set_index('geo11', inplace=True)

    return gpd.GeoDataFrame(pd.merge(acs_data, gdf,
                                          how='inner',
                                          left_index=True,
                                          right_index=True), crs=gdf.crs)

    
def get_gmean_from_neighbors(w, df, tract, column):
    """
    Get the geometric mean for a tract using the tract's neighbors
    :param w: (dictionary-like) contains the neighbors for each tract
    :param df: DataFrame with missing values to fill
    :param tract: (str) Tract that contains missing value for specified column
    :param column: (str) Column for which tract is missing value
    :return: (float) the geometric mean
    """

    #print("\tSearching for: " + ", ".join(w[tract]))
    #print()
    neighbor_values = df.loc[w.neighbors[tract], column].values
    #print("\t\tFound " + column + ": " + ", ".join(
    #    map(str, neighbor_values)))
    #print()
    geomean = \
        gmean(neighbor_values[np.logical_not(np.isnan(neighbor_values))])
    #print("\t\tMean found: ", geomean)
    #print()

    return geomean
    
    
def fillna(acs_data, state, gmean=False, shapefile=None): #Should default be True
    """
    Fill NaNs using either the geometric means of the tract's neighbors, or
    using the simple median for the column. Must include a shapefile if you want
    to use the geometric mean.
    :param acs_data: Full DataFrame
    :param state: Geography for which you wish to fill missing values
    :param gmean: (bool) If true, computes the geometric mean
    :param shapefile: (str) file path to shapefile
    :return: None. Updates data in place
    """

    if gmean:
        error_msg = "Must include a shapefile to compute the gmean"
        assert shapefile is not None, error_msg
        df = merge_geodata(acs_data, state, shapefile)
        # Computes the spatial weights used to find neighbors
        w = pysal.lib.weights.Queen.from_dataframe(df.reset_index(),
                                                   idVariable='geo11')
    else:
        df = acs_data.copy()

    nans = findna(acs_data)
    
    
    for col in nans:
        print(col)
        # Get the tracts that having missing values for the given column
        tracts = df[df[col].isna()].index.values
        for tract in tracts:
            print(tract)
            if gmean:
                print("gmean found")
                m = get_gmean_from_neighbors(w, df, tract, col)
            else:
                print("gmean not found")
                #If no gmean is included, simply fills the NaNs with the median
                m = df[col].median()
            acs_data.loc[tract, col] = m
            print("IMPPPP", len(nans) - len(findna(acs_data)))


def process_normalize(df_train, df_test, col):
    """
    Function to normalize features to have mean 0 and std 1 with training dataset
    :params df_train(pandas dataframe): Train dataset
            df_test(array): Test data
            col(list of str): Features to be normalized
    :return df_train, df_test(tuple of dataframe): Normalized datasets
    """
    scaler = StandardScaler()
    df_train[col] = scaler.fit_transform(df_train[col])
    df_test[col] = scaler.transform(df_test[col])
    
    return df_train, df_test
    
















