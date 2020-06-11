#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:01:31 2020

@author: manasip
"""
##---------------------------------------------------------------##
##              Import Libraries
##---------------------------------------------------------------##
import os


##---------------------------------------------------------------##
##              Directory and filenames
##---------------------------------------------------------------##
#Directory
wd = os.getcwd()


#Data variables
#features
ffolder = os.path.join(wd, "features_dataset")
ffilename = "_acs5_features.csv"

#Target
sfolder = os.path.join(wd, "ses_dataset")
sfilename = "_socioeconomic_vars.csv"

#Clean file
cfilename = '_features_clean_v1.1.csv'

##---------------------------------------------------------------##
##              No. of years
##---------------------------------------------------------------##
numyr = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]


##---------------------------------------------------------------##
##              Names of states, counties and shapefile
##---------------------------------------------------------------##
#Names of states
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

#Names of Counties

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


#Shapefile
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


##---------------------------------------------------------------##
##              Model setting
##---------------------------------------------------------------##

#Define models for cross validation

#model1 = {'filename':'_features_clean_v1.1.csv','numyr':['10', '11'], 'ses_input':'in_between_inputs_pca_2010_2015.csv',\
#          'ses': 'in_between_ses_2010_2015.csv'}

model1 = {'name':'model1','numyr':['10', '11'], 'ses_input':'in_between_inputs_pca_2011_2016.csv',\
          'ses': 'in_between_ses_2011_2016.csv'}

model2 = {'name':'model2','numyr':['10', '11', '12'], 'ses_input':'in_between_inputs_pca_2012_2017.csv',\
          'ses': 'in_between_ses_2012_2017.csv'}

model3 = {'name':'model3','numyr':['10', '11', '12', '13'], 'ses_input':'in_between_inputs_pca_2013_2018.csv',\
          'ses': 'in_between_ses_2013_2018.csv'}


MODELS = [model1, model2, model3]

#Test train split
tsize = 0.2

#Random seed
rseed = 20200611


dropcol_features = ['GEO_ID', 'Year','State', 'County', 'Affiliated City','Median Annual Household Income', \
           'Median Monthly Housing Costs', 'Median Value for Owner Occupied Housing Units',\
          'Percent White Collar', 'Percent College Graduate']

dropcol_input = ['per_white_collar_post', 'per_grads_post', 'median_hh_inc_log_post', \
                 'median_mhc_log_post', 'median_housing_value_log_post']
