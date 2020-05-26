"""
ACSData class for preprocessing

@author: Marc Richardson
@last updated: May 25, 2020
- Currently supports only ACS 5-year data for Cook County of Illinois
"""

# import libraries

import os
import pandas as pd
import geopandas as gpd
import numpy as np
import pysal
from scipy.stats import gmean

# System settings

wd = os.getcwd()
# Assumes a particular setup for the directories
VARS = os.path.join(wd, 'ACS', 'SES Ascent Datasets')
GEO = os.path.join(wd, 'shapefiles', 'tl_2010_17031_tract10')

# Global variables for importing ACS5 data

DTYPE = {
    'Median Annual Household Income': np.float64,
    'Median Monthly Housing Costs': np.float64,
    'Median Value for Owner Occupied Housing Units': np.float64,
    'Percent White Collar': np.float64,
    'Percent College Graduate': np.float64,
    'geo11': str
}

COL_NAMES = {
    'Median Annual Household Income': 'median_hh_inc',
    'Median Monthly Housing Costs': 'median_mhc',
    'Median Value for Owner Occupied Housing Units': 'median_housing_value',
    'Percent White Collar': 'per_white_collar',
    'Percent College Graduate': 'per_grads'
}

UNPOPULATED = ['17031381700', '17031980000', '17031980100', '17031990000']

# helper function for loading data

def convert_acs_encoding(df):

    data = df.copy()
    nan_code = -666666666

    for col in data.columns:
        indexes = data[data[col] == nan_code].index
        data.loc[indexes, col] = np.nan

    return data


class ACSData:

    def __init__(self, file_path):

        df = pd.read_csv(os.path.join(VARS, file_path), dtype=DTYPE)
        df = df[[x for x in DTYPE.keys()]]
        df.sort_values(by='geo11', inplace=True)
        df.set_index('geo11', inplace=True)
        df.drop(UNPOPULATED, inplace=True)
        df.rename(columns=COL_NAMES, inplace=True)
        df = convert_acs_encoding(df)

        self.data = df
        self.nrows, self.ncolumns = self.data.shape
        self.columns = self.data.columns
        self.index = self.data.index


    def are_tracts_same(self, other):

        return all(self.index == other.index)


    def merge_geodata(self, shapefile):

        gdf = gpd.read_file(os.path.join(GEO, shapefile))
        gdf.rename(columns={'GEOID10': 'geo11'}, inplace=True)
        gdf.set_index('geo11', inplace=True)

        return gpd.GeoDataFrame(pd.merge(self.data, gdf,
                                              how='inner',
                                              left_index=True,
                                              right_index=True), crs=gdf.crs)


    def findna(self):

        cnt = lambda x: self.data[x].isna().sum()

        return {col: cnt(col) for col in self.data.columns if cnt(col)}


    def fillna(self, gmean=False, shapefile=None):

        if gmean:
            error_msg = "Must include a shapefile to compute the gmean"
            assert shapefile is not None, error_msg
            df = self.merge_geodata(shapefile)
            w = pysal.lib.weights.Queen.from_dataframe(df.reset_index(),
                                                       idVariable='geo11')

        nans = self.findna()
        for col in nans:
            tracts = df[df[col].isna()].index.values
            for tract in tracts:
                if gmean:
                    m = self.get_gmean_from_neighbors(w, df, tract, col)
                else:
                    m = df[col].median()
                self.data.loc[tract, col] = m


    def get_gmean_from_neighbors(self, w, df, tract, column):
        """
        Find geometric mean of a tracts neighbours' for a given column.
        """

        print("\tSearching for: " + ", ".join(w[tract]))
        print()
        neighbor_values = df.loc[w.neighbors[tract], column].values
        print("\t\tFound " + column + ": " + ", ".join(
            map(str, neighbor_values)))
        print()
        geomean = \
            gmean(neighbor_values[np.logical_not(np.isnan(neighbor_values))])
        print("\t\tMean found: ", geomean)
        print()

        return geomean
