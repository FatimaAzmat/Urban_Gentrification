# Predicting Neighborhood Gentrification in the U.S.
## CAPP 30254 | Spring 2020

## Project Members:  
Fatima Azmat  
Launa Greer  
Manasi Phadnis  
Marc Richardson

## Summary
We implement a machine learning (ML) model that (1) produces a socioeconomic status score for each neighborhood (as defined at the census tract level) across 24 different U.S. cities and (2) predicts which neighborhoods are likely to experience socioeconomic ascent over a five-year period, signifying the potential risk of gentrification. Our model builds on existing efforts to apply machine learning methods to the study of neighborhood change (Greene & Pettit (2016), (Reades et al. (2019), (Knorr (2019)). To generate both our feature data and our target data, we utilize demographic and socioeconomic data from the five-year American Community Survey collected by the U.S. Census Bureau. We use principal component analysis (PCA) to generate a relative socioeconomic score for each neighborhood and take the change in that score over four five-year periods from 2010 to 2018 to identify ascending and declining neighborhoods. We then use Random Forest regression to predict the socioeconomic ascent (or decline) of a neighborhood based on historical feature data. Our best performing model with hyperparameter tuning has a Mean Squared Error of 65.27, Mean Absolute Error of 5.86 and Explained Variance of 0.17. The intended use of our model is to assist local and state governments more effectively advocate for and achieve equitable, inclusive outcomes that result from neighborhood change. We also seek to expand efforts to establish a sustainable neighborhood early detection system for local communities.

## Organization
acs
> config
    - master: JSON file specifying the ACS source type (e.g., "acs5"), cities, and years for which to retrieve data
    - shpfiles: state shapefiles
    - variables: ACS variables to retrieve for a given publication year
    - geos.json: JSON file containing names and FIPS codes for a city and its affiliated state and counties

> outputs
    - Contains CSV files generated as the output of dataretrieval.py
    - Uncleaned feature data is saved as "acs5_yyyy_features.csv"
    - Socioeconomic data for the PCA analysis is saved as "acs5_yyyy_socioeconomic_vars.csv"

> acsclient.py:
    - Retrieves ACS data using the "CensusData" Python package and/or the Census Bureau's direct API endpoint

> dataretrieval.py:
    - Orchestrates the retrieval of ACS data
    - Generates model features and the input variables used in the PCA analysis as two CSV files

> models.py:
    - Holds classes representing ACS geographies and data requests

common
> vishelper.py
    - Provides methods for creating data visualizations

model
> clean.py
    - Orchestrates the cleaning of the feature data
> process.py
    - A collection of various cleaning functions, such as finding nan values, imputing the geometric mean, etc.
> userinput.py
    - A collection of input/configuration variables to use for the cleaning

notebooks
> A collection of Jupyter notebooks to demonstrate various functionalities of the data pipeline and model creation

pca
> dataset.py
    - Defines methods and classes to facilitate the cleaning of socioeconomic data in preparation for the PCA analysis
> pca.sh
    - An shell script to generate outcome variables for each year while applying one of three transformations to the 
> pca_analysis.py
    - Runs a principal component analysis on the socioeconomic variables provided as a CSV file by acs/dataretrieval.py
    - Following the PCA analysis, generates a socioeconomic score for that year's data
    - Takes the difference between socioeconomic scores for each census tract to compute that tract's "socioeconomic ascent," i.e., the outcome variable

## Instructions
(1) Clone code from repository.
(2) Retrieve ACS data and create variables for PCA analysis and feature set by running dataretrieval.py.
(3) Run PCA analysis and generate outcome variables by running pca.sh.
(4) Clean features.
(5) Train and test random forest regression model using available notebooks.
