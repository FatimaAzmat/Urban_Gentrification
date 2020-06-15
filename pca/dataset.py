"""
ACSData class for PCA pre-processing

@author: Marc Richardson
@last updated: June 8, 2020
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
SF = os.path.join(wd, 'shapefiles')
GEO_IL = os.path.join(SF, 'tl_2010_17_tract10')
GEO_DC = os.path.join(SF, 'tl_2010_11_tract10')
GEO_NY = os.path.join(SF, 'tl_2010_36_tract10')
GEO_CA = os.path.join(SF, 'tl_2010_06_tract10')
GEO_WA = os.path.join(SF, 'tl_2010_53_tract10')
GEO_PA = os.path.join(SF, 'tl_2010_42_tract10')
GEO_OR = os.path.join(SF, 'tl_2010_41_tract10')
GEO_GA = os.path.join(SF, 'tl_2010_13_tract10')
GEO_MN = os.path.join(SF, 'tl_2010_27_tract10')
GEO_LA = os.path.join(SF, 'tl_2010_22_tract10')
GEO_NM = os.path.join(SF, 'tl_2010_35_tract10')
GEO_OK = os.path.join(SF, 'tl_2010_40_tract10')
GEO_TX = os.path.join(SF, 'tl_2010_48_tract10')
GEO_MA = os.path.join(SF, 'tl_2010_25_tract10')
GEO_NJ = os.path.join(SF, 'tl_2010_34_tract10')
GEO_MD = os.path.join(SF, 'tl_2010_24_tract10')
GEO_MI = os.path.join(SF, 'tl_2010_26_tract10')
GEO_FL = os.path.join(SF, 'tl_2010_12_tract10')
GEO_NC = os.path.join(SF, 'tl_2010_37_tract10')

# Global variables for importing ACS5 data

DTYPE = {
    'Median Annual Household Income': np.float64,
    'Median Monthly Housing Costs': np.float64,
    'Median Value for Owner Occupied Housing Units': np.float64,
    'Percent White Collar': np.float64,
    'Percent College Graduate': np.float64,
    'Total Population': np.float64,
    'geo11': str,
    'State': str,
    'County': str,
    'Affiliated City': str
}

COL_NAMES = {
    'Median Annual Household Income': 'median_hh_inc',
    'Median Monthly Housing Costs': 'median_mhc',
    'Median Value for Owner Occupied Housing Units': 'median_housing_value',
    'Percent White Collar': 'per_white_collar',
    'Percent College Graduate': 'per_grads'
}

SF_DICT = {
    'District of Columbia': GEO_DC,
    'Washington': GEO_WA,
    'California': GEO_CA,
    'New York': GEO_NY,
    'Illinois': GEO_IL,
    'Pennsylvania': GEO_PA,
    'Oregon': GEO_OR,
    'Georgia': GEO_GA,
    'Minnesota': GEO_MN,
    'Louisiana': GEO_LA,
    'New Mexico': GEO_NM,
    'Oklahoma': GEO_OK,
    'Texas': GEO_TX,
    'Massachusetts': GEO_MA,
    'New Jersey': GEO_NJ,
    'Maryland': GEO_MD,
    'Michigan': GEO_MI,
    'Florida': GEO_FL,
    'North Carolina': GEO_NC
}

COUNTIES = {'California': ['Los Angeles',
                           'San Diego',
                           'San Francisco County'],
            'District of Columbia': ['District of Columbia'],
            'Florida': ['Miami-Dade'],
            'Georgia': ['Fulton'],
            'Illinois': ['Cook County',
                         'DuPage County'],
            'Louisiana': ['Orleans'],
            'Maryland': ['Baltimore City'],
            'Massachusetts': ['Suffolk'],
            'Michigan': ['Wayne'],
            'Minnesota': ['Hennepin'],
            'New Jersey': ['Hudson'],
            'New Mexico': ['Bernalillo'],
            'New York': ['Bronx County',
                         'Kings County',
                         'New York County',
                         'Queens County',
                         'Richmond County'],
            'North Carolina': ['Mecklenburg'],
            'Oklahoma': ['Tulsa'],
            'Oregon': ['Multnomah'],
            'Pennsylvania': ['Allegheny',
                             'Philadelphia'],
            'Texas': ['Travis',
                      'Harris'],
            'Washington': ['King County']
            }

# helper function for loading data

def convert_acs_encoding(df):

    data = df.copy()
    nan_code = -666666666

    for col in data.columns:
        indexes = data[data[col] == nan_code].index
        data.loc[indexes, col] = np.nan

    return data


def find_unpopulated_tracts(df):
    """
    Drop unpopulated tracts
    """

    data = df.copy()
    tracts = data[data['Total Population'] == 0].index
    print('Dropping unpopulated tracts: {}\n'.format(tracts))

    return tracts


class ACSData:

    def __init__(self, file_path):

        df = pd.read_csv(os.path.join(VARS, file_path), dtype=DTYPE)
        df = df[[x for x in DTYPE.keys()]]
        df.sort_values(by='geo11', inplace=True)
        df.set_index('geo11', inplace=True)
        df.rename(columns=COL_NAMES, inplace=True)
        df = convert_acs_encoding(df)
        df.drop(find_unpopulated_tracts(df), inplace=True)

        self.data = df
        self.nrows, self.ncolumns = self.data.shape
        self.columns = self.data.columns
        self.index = self.data.index


    def make_tracts_same(self, other):
        """
        Forces data to have same tracts by removing tracts that appear
        in one data set but not the other
        """

        diff1 = set(self.index) - set(other.index)
        diff2 = set(other.index) - set(self.index)

        if diff1:
            print(f"Dropping {len(diff1)} tracts from first data set")
            self.data.drop(diff1, inplace=True)

        if diff2:
            print(f"Dropping {len(diff2)} tracts from second data set")
            other.data.drop(diff2, inplace=True)

        self.nrows, self.ncolumns = self.data.shape
        self.index = self.data.index

        other.nrows, other.ncolumns = other.data.shape
        other.index = other.data.index


    def are_tracts_same(self, other):

        try:
            match = all(self.index == other.index)
        except ValueError:
            match = False

        return match


    def get_pca_vars(self, exclude_mhc=False):

        if exclude_mhc:
            cols = \
                [x for x in self.columns if x in COL_NAMES.values() \
                 and x != 'median_mhc']
        else:
            cols = [x for x in self.columns if x in COL_NAMES.values()]

        return self.data[cols]


    def merge_geodata(self, state, county):

        dir_ = SF_DICT[state]
        shapefile = dir_[dir_.rindex('/') + 1:] + '.shp'
        gdf = gpd.read_file(os.path.join(dir_, shapefile))
        gdf.rename(columns={'GEOID10': 'geo11'}, inplace=True)
        gdf.set_index('geo11', inplace=True)
        df = self.data.copy()
        df = df[df['County'] == county]

        return gpd.GeoDataFrame(pd.merge(df, gdf,
                                         how='inner',
                                         left_index=True,
                                         right_index=True), crs=gdf.crs)


    def findna(self):

        cnt = lambda x: self.data[x].isna().sum()

        return {col: cnt(col) for col in self.data.columns if cnt(col)}


    def fillna(self, state, county):

        df = self.merge_geodata(state, county)
        w = pysal.lib.weights.Queen.from_dataframe(df.reset_index(),
                                                   idVariable='geo11')

        nans = self.findna()
        # print(f"\tFilling missing values for {county} in {state}\n")
        for col in nans:
            tracts = df[df[col].isna()].index.values
            if not len(tracts):
                # print('\t\tNo tracts with missing values\n')
                continue
            # print('\t\tFilling missing values in {} for {} tracts\n'.format(col, len(tracts)))
            # print('\t\tTracts: {}\n'.format(tracts))
            for tract in tracts:
                m = self.get_gmean_from_neighbors(w, df, tract, col)
                self.data.loc[tract, col] = m


    def get_gmean_from_neighbors(self, w, df, tract, column):
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
            # print(f'\t\t\t\tNo neighbors found for {tract}... Using county median\n')
            geomean = df[column].median()
        # print("\t\t\t\tMean found: ", geomean)
        # print()

        return geomean


    def impute_missing_values(self):

        for state, counties in COUNTIES.items():
            for county in counties:
                self.fillna(state, county)
