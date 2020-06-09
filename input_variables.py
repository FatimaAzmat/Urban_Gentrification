#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 00:03:51 2020

@author: manasip
"""

##Module to define global variables. User to change input

import os
#
## System settings
wd = os.getcwd()


## Assumes a particular setup for the directories (PCA/shapefiles/tl_2010_XX_tract10/shapefile)
SF = os.path.join(wd, 'shapefiles')

##State wise tracts folder
GEO_IL = os.path.join(SF, 'tl_2010_17_tract10')
GEO_DC = os.path.join(SF, 'tl_2010_11_tract10')
GEO_NY = os.path.join(SF, 'tl_2010_36_tract10')
GEO_CA = os.path.join(SF, 'tl_2010_06_tract10')
GEO_WA = os.path.join(SF, 'tl_2010_53_tract10')

##Create dictionary for states. States being used are Illinois, District of Columbia, New York, California, Washington
SF_DICT = {
    'District of Columbia': GEO_DC,
    'Washington': GEO_WA,
    'New York': GEO_NY,
    'Illinois': GEO_IL,
    'California': GEO_CA
}


#State wise shape files
STATE_SF = {
    'District of Columbia': 'tl_2010_11_tract10.shp',
    'Washington': 'tl_2010_53_tract10.shp',
    'New York': 'tl_2010_36_tract10.shp',
    'Illinois': 'tl_2010_17_tract10.shp',
    'California': 'tl_2010_06_tract10.shp'
}

#Define number of years of features data
numyr = ['10', '11']


#Define train test split
tsize = 0.2

#Define value of seed
rseed = 20200611

#Outcome file
outcome = "untransformed_pca_ses_2010_2015.csv"

#Features data check why this doesnt work
fname = "../features_dataset"
ffname = "_acs5_features.csv"

#Drop columns
dropcol = ['GEO_ID', 'Year','State', 'County', 'Affiliated City','Median Annual Household Income', \
           'Median Monthly Housing Costs', 'Median Value for Owner Occupied Housing Units',\
          'Percent White Collar', 'Percent College Graduate']


 
