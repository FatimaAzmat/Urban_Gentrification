"""
PCA Analysis
@author: Marc Richardson
@last updated: June 3, 2020

Does the PCA analysis on the two datasets, saves the input and output to a csv
"""

# import libraries

import os, sys
import pandas as pd
import numpy as np
import random
import argparse

from sklearn.decomposition import PCA
from sklearn import preprocessing

from scipy.stats import boxcox

# System settings

wd = os.getcwd()
# Assumes a particular setup for the directories
VARS = os.path.join(wd, 'ACS', 'SES Ascent Datasets')
SF = os.path.join(wd, 'shapefiles')
# GEO_IL = os.path.join(SF, 'tl_2010_17_tract10')
# GEO_DC = os.path.join(SF, 'tl_2010_11_tract10')
# GEO_NY = os.path.join(SF, 'tl_2010_36_tract10')
# GEO_CA = os.path.join(SF, 'tl_2010_06_tract10')
# GEO_WA = os.path.join(SF, 'tl_2010_53_tract10')
OUTPUT = os.path.join(wd, 'scores')

sys.path.insert(0, VARS)
# sys.path.insert(1, GEO_IL)
# sys.path.insert(2, GEO_DC)
# sys.path.insert(3, GEO_NY)
# sys.path.insert(4, GEO_CA)
# sys.path.insert(5, GEO_WA)
sys.path.insert(1, OUTPUT)

# Local modules

import dataset as ds
from dataset import SF_DICT

# Set random seed for replicability

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

# Global variables

ACS_FILE = 'acs5_XXXX_socioeconomic_vars.csv'

VARS = ['per_white_collar',
        'per_grads',
        'median_housing_value',
        'median_hh_inc',
        'median_mhc']

FIPS = {
    'District of Columbia': '11',
    'Washington': '53',
    'California': '06',
    'New York': '36',
    'Illinois': '17',
    'Pennsylvania': '42',
    'Oregon': '41',
    'Georgia': '13',
    'Minnesota': '27',
    'Louisiana': '22',
    'New Mexico': '35',
    'Oklahoma': '40',
    'Texas': '48',
    'Massachusetts': '25',
    'New Jersey': '34',
    'Maryland': '24',
    'Michigan': '26',
    'Florida': '12',
    'North Carolina': '37'
}

# Helper functions for main

def transform(columns, df1, df2, transformation='in_between'):

    df1 = df1.copy()
    df2 = df2.copy()

    if transformation == 'box_cox':

        for col in columns:
            values1, lmd = boxcox(df1[col])
            values2 = boxcox(df2[col], lmbda=lmd)
            df1.loc[:, col + '_box'] = values1
            df2.loc[:, col + '_box'] = values2
            df1.drop([col], axis=1, inplace=True)
            df2.drop([col], axis=1, inplace=True)

    elif transformation == 'in_between':

        for col in columns:
            if col == 'median_hh_inc':
                df1.loc[:, col + '_log'] = np.power(df1[col], 2.0/3.0)
                df2.loc[:, col + '_log'] = np.power(df2[col], 2.0/3.0)
            else:
                df1.loc[:, col + '_log'] = np.log(df1[col])
                df2.loc[:, col + '_log'] = np.log(df2[col])

            df1.drop([col], axis=1, inplace=True)
            df2.drop([col], axis=1, inplace=True)

    return df1, df2


def do_pca(df1, df2):

    df1 = df1.copy()
    df2 = df2.copy()

    array = np.concatenate((df1, df2), axis=0)

    assert np.isfinite(array).any(), 'Error: array contains infinite values'
    assert ~np.isnan(array).any(), 'Error: array contains NaN values'

    scaler = preprocessing.RobustScaler()
    scaler.fit(array)
    array = scaler.transform(array)

    pca = PCA(n_components=1)
    pca.fit(array)
    scores = pd.DataFrame(pca.transform(array))
    print(
        "The amount of explained variance of the SES score is: {0:6.5f}".format(
            pca.explained_variance_ratio_[0]))

    scores1 = scores.loc[:len(df1) - 1, 0]
    scores2 = scores.loc[len(df1):, 0]

    df1 = df1.assign(scores=pd.Series(scores1).values)
    df2 = df2.assign(scores=pd.Series(scores2).values)

    df = df1.merge(df2, how='outer', suffixes=('_pre', '_post'),
                   left_index=True, right_index=True)

    return df


def compute_rank_ascent(df):

    df = df.copy()

    df.loc[:, 'rank_pre'] = df['scores_pre'].rank(ascending=False)
    df.loc[:, 'rank_post'] = df['scores_post'].rank(ascending=False)
    df.loc[:, 'scores_asc'] = df.loc[:, 'scores_post'] - df.loc[:, 'scores_pre']
    df.loc[:, 'scores_pr_pre'] = \
        df['rank_pre'].rank(ascending=False, pct=True) * 100
    df.loc[:, 'scores_pr_post'] = \
        df['rank_post'].rank(ascending=False, pct=True) * 100
    df.loc[:, 'scores_pr_asc'] = df.loc[:, 'scores_pr_post'] - \
        df.loc[:, 'scores_pr_pre']

    inp = df.loc[:, \
          [x for x in df.columns if 'score' not in x and 'rank' not in x]]

    scores = df.loc[:, \
             [x for x in df.columns if 'score' in x or 'rank' in x]]

    return inp, scores


if __name__ == "__main__":

    # Parse command line arguments

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--begin", help="Input starting year", required=True, type=int)
    parser.add_argument(
        "-e", "--end", help="Input ending year", required=True, type=int)
    # parser.add_argument(
    #     "-g", "--geometry",
    #     help="Use spatial analysis to compute missing values",
    #     action='store_true')
    parser.add_argument("-t", "--transform", help="Input transformation",
                        required=True, type=str)
    args = parser.parse_args()

    # Check that arguments are valid

    # print(args.begin, args.end, args.transform)
    assert args.transform in ['box_cox', 'in_between', 'untransformed'], \
        'Invalid transformation'
    assert args.begin in range(2010, 2019), 'Invalid beginning year'
    assert args.end in range(2010, 2019), 'Invalid ending year'
    assert args.begin < args.end, 'Beginning year must be before ending year'

    print("Loading datasets for {} and {}...\n".format(args.begin, args.end))

    dataSet1 = ds.ACSData(ACS_FILE.replace('XXXX', str(args.begin)))
    dataSet2 = ds.ACSData(ACS_FILE.replace('XXXX', str(args.end)))

    print("Looking for missing values and imputing value...\n")

    dataSet1.impute_missing_values()
    dataSet2.impute_missing_values()

    # if args.geometry:

        # for state, sf in SF_DICT.items():
        #     print("Filling values for {}\n".format(state))
        #     print("\t" + str(args.begin) + "...\n")
        #     dataSet1.fillna(state, gmean=True, shapefile=sf)
        #     print("\t" + str(args.end) + "...\n")
        #     dataSet2.fillna(state, gmean=True, shapefile=sf)
    # else:
    #     dataSet1.fillna()
    #     dataSet2.fillna()

    if not dataSet1.are_tracts_same(dataSet2):
        print('Warning: the two datasets have sets of tracts that do not match'
              '\tDropping dissimilar tracts datasets...')
        dataSet1.make_tracts_same(dataSet2)

    print('Gathering columns for PCA')

    df1 = dataSet1.get_pca_vars()
    df2 = dataSet2.get_pca_vars()

    print('Transforming data using specified transformation...\n')

    transformed = transform([x for x in df1.columns if 'median' in x], df1, df2,
                            transformation=args.transform)

    print('Doing PCA analysis...\n')

    scores = do_pca(transformed[0], transformed[1])

    inputs, output = compute_rank_ascent(scores)

    print('Saving inputs and output to files...\n')

    inputs_file = (str(args.transform) + '_inputs_pca_' + str(args.begin) +
                   '_' + str(args.end) + '.csv')
    output_file = (str(args.transform) + '_ses_' + str(args.begin) + '_' +
                   str(args.end) + '.csv')

    if not os.path.exists(OUTPUT):
        os.mkdir(OUTPUT)

    inputs.to_csv(os.path.join(OUTPUT, inputs_file), index=True)
    output.to_csv(os.path.join(OUTPUT, output_file), index=True)

    print('Done')
