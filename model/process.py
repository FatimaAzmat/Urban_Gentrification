#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:01:17 2020

@author: manasip
"""


import numpy as np
import geopandas as gpd
import os
import pandas as pd
import pysal
from scipy.stats import gmean
from sklearn.preprocessing import StandardScaler

from userinput import SF_DICT, COUNTIES, SF, GEO_IL, GEO_DC, GEO_NY, GEO_CA, GEO_WA, GEO_PA, GEO_OR, GEO_GA, GEO_MN, \
GEO_LA, GEO_NM, GEO_OK, GEO_TX, GEO_MA, GEO_NJ, GEO_MD, GEO_MI, GEO_FL, GEO_NC



def chg_col(acs_data, colname="geo11"):
    """
    Function sorts by column name, changes column type to str and sets column as index
    :params df(pandas dataframe): ACS data
            colname(str): Column to be formatted
    """
    acs_data.sort_values(by=colname, inplace=True)
    acs_data[colname] = acs_data[colname].astype(str)
    acs_data.set_index(colname, inplace=True)


def convert_acs_encoding(acs_data):

    data = acs_data.copy()
    nan_code = -666666666.0

    for col in data.columns:
        indexes = data[data[col] == nan_code].index
        data.loc[indexes, col] = np.nan

    return data


def find_unpopulated_tracts(acs_data):
    """
    Drop unpopulated tracts
    """

    data = acs_data.copy()
    tracts = data[data['Total Population'] == 0].index
    print('Dropping unpopulated tracts: {}\n'.format(tracts))

    return tracts


def merge_geodata(acs_data, state, county):

        dir_ = SF_DICT[state]
        shapefile = dir_[dir_.rindex('/') + 1:] + '.shp'
        gdf = gpd.read_file(os.path.join(dir_, shapefile))
        gdf.rename(columns={'GEOID10': 'geo11'}, inplace=True)
        gdf.set_index('geo11', inplace=True)
        df = acs_data.copy()
        df = df[df['County'] == county]

        return gpd.GeoDataFrame(pd.merge(df, gdf,
                                         how='inner',
                                         left_index=True,
                                         right_index=True), crs=gdf.crs)

def findna(acs_data):

        cnt = lambda x: acs_data[x].isna().sum()

        return {col: cnt(col) for col in acs_data.columns if cnt(col)}
    

def fillna(acs_data, state, county):

        df = merge_geodata(acs_data, state, county)
        w = pysal.lib.weights.Queen.from_dataframe(df.reset_index(),
                                                   idVariable='geo11')

        nans = findna(acs_data)
        print(f"\tFilling missing values for {county} in {state}\n")
        for col in nans:
            tracts = df[df[col].isna()].index.values
            if not len(tracts):
                print('\t\tNo tracts with missing values\n')
                continue
            print('\t\tFilling missing values in {} for {} tracts\n'.format(col, len(tracts)))
            print('\t\tTracts: {}\n'.format(tracts))
            for tract in tracts:
                m = get_gmean_from_neighbors(w, df, tract, col)
                acs_data.loc[tract, col] = m
                
                
                
def get_gmean_from_neighbors(w, df, tract, column):
        """
        Find geometric mean of a tracts neighbours' for a given column.
        """

        # print("\t\t\tSearching for: " + ", ".join(w[tract]))
        # print()
        neighbor_values = df.loc[w.neighbors[tract], column].values
        # print("\t\t\tFound " + column + ": " + ", ".join(
        #     map(str, neighbor_values)))
        # print()
        if not all(np.isnan(neighbor_values)):
            # print('\t\t\t\tUsing geometric mean\n')
            geomean = \
                gmean(neighbor_values[np.logical_not(np.isnan(neighbor_values))])
        else:
            print(f'\t\t\t\tNo neighbors found for {tract}... Using county median\n')
            geomean = df[column].median()
        # print("\t\t\t\tMean found: ", geomean)
        # print()

        return geomean


def impute_missing_values(acs_data):
    for state, counties in COUNTIES.items():
        for county in counties:
            fillna(acs_data, state, county)
            
            
        
def process_normalize(df_train, df_test, col):
    '''
    Function to normalize features to have mean 0 and std 1 with training dataset
    
    Input
        df_train(pandas dataframe): Train dataset
        df_test(pandas dataframe): Test dataset
        col(list of str): Features to be normalized
    
    Output
        df_train, df_test(tuple of dataframe): Normalized datasets
    '''
    scaler = StandardScaler()
    df_train[col] = scaler.fit_transform(df_train[col])
    df_test[col] = scaler.transform(df_test[col])
    
    return df_train, df_test
