"""
PCA Analysis
@author: Marc Richardson
@last updated: June 8, 2020

Does the PCA analysis on the two datasets, saves the inputs and outputs to a csv
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
OUTPUT = os.path.join(wd, 'scores')
BY_CITY = os.path.join(wd, 'city_scores')
RELATIVE_SCORES = os.path.join(wd, 'relative_rank')
LOADINGS = os.path.join(wd, 'loadings')
CITY_LOADINGS = os.path.join(LOADINGS, 'cities')
DROP_MHC = os.path.join(wd, 'without_mhc')

sys.path.insert(0, VARS)
sys.path.insert(1, OUTPUT)

# Local modules

import dataset as ds

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

    # Median removal and unit scaling

    scaler = preprocessing.RobustScaler()
    scaler.fit(array)
    array = scaler.transform(array)

    # Generate loadings

    pca_full = PCA()
    pca_full.fit(array)

    i = np.identity(array.shape[1])
    coef = pca_full.transform(i)
    components = coef.shape[1]
    explained_variance = \
        pca_full.explained_variance_ratio_.reshape(1, components)
    loadings = np.concatenate((coef, explained_variance), axis=0)
    loadings = pd.DataFrame(loadings,
                            index=list(df1.columns) + ['Explained Variance'])

    # Get first component for SES

    pca = PCA(n_components=1)
    pca.fit(array)
    scores = pd.DataFrame(pca.transform(array))

    # Disaggregate scores into separate years

    scores1 = scores.loc[:len(df1) - 1, 0]
    scores2 = scores.loc[len(df1):, 0]

    # Add scores to original data

    df1 = df1.assign(scores=pd.Series(scores1).values)
    df2 = df2.assign(scores=pd.Series(scores2).values)

    df = df1.merge(df2, how='outer', suffixes=('_pre', '_post'),
                   left_index=True, right_index=True)

    return df, loadings


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

    # Get inputs used in PCA

    inp = df.loc[:, \
          [x for x in df.columns if 'score' not in x and 'rank' not in x]]

    # Get scores and ranks

    scores = df.loc[:, \
             [x for x in df.columns if 'score' in x or 'rank' in x]]

    return inp, scores


if __name__ == "__main__":

    # Parse command line arguments

    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--begin", help="Input starting year",
                        required=True, type=int)

    parser.add_argument("-e", "--end", help="Input ending year",
                        required=True, type=int)

    parser.add_argument("-t", "--transform", help="Input transformation",
                        required=True, type=str)

    parser.add_argument("-r", "--relative", help="Generate relative ranks",
                        default=False, action="store_true")

    parser.add_argument("-c", "--city", help="Generate scores within cities",
                        default=False, action="store_true")

    parser.add_argument("-x", "--exclude", help="Exclude monthly housing cost",
                        default=False, action="store_true")

    args = parser.parse_args()

    # Check that arguments are valid

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

    if not dataSet1.are_tracts_same(dataSet2):
        print('Warning: the two datasets have sets of tracts that do not match'
              '\tDropping dissimilar tracts datasets...')
        dataSet1.make_tracts_same(dataSet2)

    print('Gathering columns for PCA...\n')

    if args.exclude:
        df1 = dataSet1.get_pca_vars(exclude_mhc=True)
        df2 = dataSet2.get_pca_vars(exclude_mhc=True)
    else:
        df1 = dataSet1.get_pca_vars()
        df2 = dataSet2.get_pca_vars()

    print('Transforming data using specified transformation...\n')

    transformed = transform([x for x in df1.columns if 'median' in x], df1, df2,
                            transformation=args.transform)

    print('Doing PCA analysis...\n')

    if not os.path.exists(LOADINGS):
        os.mkdir(LOADINGS)

    if args.city:

        if not os.path.exists(CITY_LOADINGS):
            os.mkdir(CITY_LOADINGS)

        scores = pd.DataFrame()

        cities = dataSet1.data.groupby('Affiliated City').groups.items()

        for city, idxs in cities:
            if city == "Austin":
                continue
            city_pre, city_post = transformed
            city_pre = city_pre.loc[idxs, :]
            city_post = city_post.loc[idxs, :]
            city_scores, loadings = do_pca(city_pre, city_post)
            scores = pd.concat((scores, city_scores))
            loadings_file = os.path.join(CITY_LOADINGS, str(args.transform) +
                                         '_' + city + '_' + 'loadings' + '_' +
                                         str(args.begin) + '_' + str(args.end)
                                         + '.csv')
            loadings.to_csv(loadings_file)

    else:

        scores, loadings = do_pca(transformed[0], transformed[1])

        if not args.exclude:
            loadings_file = os.path.join(LOADINGS, str(args.transform) +
                                         '_loadings_' + str(args.begin) +
                                         '_' + str(args.end) + '.csv')
            loadings.to_csv(loadings_file)

    print("Computing SES rank, percent rank, ascent, and percent ascent...\n")

    if args.relative:

        inputs = pd.DataFrame()
        output = pd.DataFrame()

        for idxs in dataSet1.data.groupby('Affiliated City').groups.values():
            by_city = scores.loc[idxs, :]
            input_, output_ = compute_rank_ascent(by_city)
            inputs = pd.concat((inputs, input_))
            output = pd.concat((output, output_))

    else:

        inputs, output = compute_rank_ascent(scores)

    print('Saving inputs and output to files...\n')

    inputs_file = (str(args.transform) + '_inputs_pca_' + str(args.begin) +
                   '_' + str(args.end) + '.csv')
    output_file = (str(args.transform) + '_ses_' + str(args.begin) + '_' +
                   str(args.end) + '.csv')

    if args.city:

        if not os.path.exists(BY_CITY):
            os.mkdir(BY_CITY)

        inputs.to_csv(os.path.join(BY_CITY, inputs_file), index=True)
        output.to_csv(os.path.join(BY_CITY, output_file), index=True)

    if args.relative:

        if not os.path.exists(RELATIVE_SCORES):
            os.mkdir(RELATIVE_SCORES)

        inputs.to_csv(os.path.join(RELATIVE_SCORES, inputs_file), index=True)
        output.to_csv(os.path.join(RELATIVE_SCORES, output_file), index=True)

    if not args.relative and not args.city and not args.exclude:

        if not os.path.exists(OUTPUT):
            os.mkdir(OUTPUT)

        inputs.to_csv(os.path.join(OUTPUT, inputs_file), index=True)
        output.to_csv(os.path.join(OUTPUT, output_file), index=True)

    if args.exclude:

        if not os.path.exists(DROP_MHC):
            os.mkdir(DROP_MHC)

        loadings_dir = os.path.join(DROP_MHC, 'loadings')

        if not os.path.exists(loadings_dir):
            os.mkdir(loadings_dir)

        loadings_file = os.path.join(loadings_dir, str(args.transform) +
                                     '_loadings_' + str(args.begin) +
                                     '_' + str(args.end) + '.csv')

        inputs.to_csv(os.path.join(DROP_MHC, inputs_file), index=True)
        output.to_csv(os.path.join(DROP_MHC, output_file), index=True)
        loadings.to_csv(loadings_file)


    print('Done ######################################################\n\n\n')
