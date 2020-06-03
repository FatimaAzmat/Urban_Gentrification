'''
dataretrieval.py

Orchestrates the collection of ACS variables related to gentrification.
'''

import acsclient
import censusdata as census
import csv
import json
import jsons
import pandas as pd
import logging

from models import GeoMapping, ACSDataRequest
from typing import List


def _compute_percent_college_graduate(df, compute_by_sex=False):
    '''
    Adds a new column to the DataFrame to capture the percentage of census
    tract residents who hold an associate's degree or higher. Removes columns
    used as intermediaries in the calculation.

    Parameters:
        df (pd.DataFrame): the DataFrame after its population from Census API data
        compute_by_sex (df): a boolean indicating whether educational attainment
                             should be aggregated by sex.  Necessary due to the
                             different ways in which educational attainment
                             was stored in the ACS 5-year survey from 2010-2011
                             compared to 2012+. Both approaches use a universe
                             of residents aged 25 and older. Defaults to False.

    Returns:
        (pd.DataFrame): a modified copy of the original DataFrame
    '''
    if compute_by_sex:
        cols = [
            "Educational Attainment: Total Male Surveyed",
            "Educational Attainment: Number Male Associates",
            "Educational Attainment: Number Male Bachelors",
            "Educational Attainment: Number Male Masters",
            "Educational Attainment: Number Male Professional School",
            "Educational Attainment: Number Male Doctorates",
            "Educational Attainment: Total Female Surveyed",
            "Educational Attainment: Number Female Associates",
            "Educational Attainment: Number Female Bachelors",
            "Educational Attainment: Number Female Masters",
            "Educational Attainment: Number Female Professional School",
            "Educational Attainment: Number Female Doctorates"
        ]
        df["Percent College Graduate"] = (
            (df["Educational Attainment: Number Male Associates"] +
            df["Educational Attainment: Number Male Bachelors"] +
            df["Educational Attainment: Number Male Masters"] +
            df["Educational Attainment: Number Male Professional School"] +
            df["Educational Attainment: Number Male Doctorates"] +
            df["Educational Attainment: Number Female Associates"] +
            df["Educational Attainment: Number Female Bachelors"] +
            df["Educational Attainment: Number Female Masters"] +
            df["Educational Attainment: Number Female Professional School"] +
            df["Educational Attainment: Number Female Doctorates"]) /
            (df["Educational Attainment: Total Male Surveyed"] +
            df["Educational Attainment: Total Female Surveyed"] )
        )
    else:
        cols = [
            "Educational Attainment: Total Surveyed",
            "Educational Attainment: Number Associates",
            "Educational Attainment: Number Bachelors",
            "Educational Attainment: Number Masters",
            "Educational Attainment: Number Professional School",
            "Educational Attainment: Number Doctorates"
        ]
        df["Percent College Graduate"] = (
            (df["Educational Attainment: Number Associates"] +
            df["Educational Attainment: Number Bachelors"] +
            df["Educational Attainment: Number Masters"] +
            df["Educational Attainment: Number Professional School"] +
            df["Educational Attainment: Number Doctorates"]) /
            df["Educational Attainment: Total Surveyed"]     
        )

    return df.drop(columns=cols)


def _compute_percent_commute_time(df):
    '''
    Adds a new column to the DataFrame to capture the percentage of census
    tract residents with different work commute times. Removes columns
    used as intermediaries in the calculation.

    Parameters:
        df (pd.DataFrame): the DataFrame after its population from Census API data

    Returns:
        (pd.DataFrame): a modified copy of the original DataFrame
    '''
    df["Percent Work Commute Under 5 Minutes"] = (
        df["Work Commute: Less than 5 minutes"] / 
        df["Work Commute: Total Surveyed"]
    )
    df["Percent Work Commute 5-14 Minutes"] = (
        (df["Work Commute: 5-9 minutes"] +
        df["Work Commute: 10-14 minutes"]) /
        df["Work Commute: Total Surveyed"]     
    )
    df["Percent Work Commute 15-29 Minutes"] = (
        (df["Work Commute: 15-19 minutes"] +
        df["Work Commute: 20-24 minutes"] +
        df["Work Commute: 25-29 minutes"]) /
        df["Work Commute: Total Surveyed"]     
    )
    df["Percent Work Commute 30-59 Minutes"] = (
        (df["Work Commute: 30-34 minutes"] +
        df["Work Commute: 35-39 minutes"] +
        df["Work Commute: 40-44 minutes"] + 
        df["Work Commute: 45-59 minutes"]) /
        df["Work Commute: Total Surveyed"]     
    )
    df["Percent Work Commute 60+ Minutes"] = (
        (df["Work Commute: 60-89 minutes"] +
        df["Work Commute: 90 or more minutes"]) /
        df["Work Commute: Total Surveyed"]     
    )

    return df.drop(columns=[
        "Work Commute: Total Surveyed",
        "Work Commute: Less than 5 minutes",
        "Work Commute: 5-9 minutes",
        "Work Commute: 10-14 minutes",
        "Work Commute: 15-19 minutes",
        "Work Commute: 20-24 minutes",
        "Work Commute: 25-29 minutes",
        "Work Commute: 30-34 minutes",
        "Work Commute: 35-39 minutes",
        "Work Commute: 40-44 minutes",
        "Work Commute: 45-59 minutes",
        "Work Commute: 60-89 minutes",
        "Work Commute: 90 or more minutes"
    ])


def _compute_percent_commute_type(df):
    '''
    Adds new columns to the DataFrame to capture percentages for different means
    of transportation to work. Removes columns used as intermediaries in
    the calculation.

    Parameters:
        df (pd.DataFrame): the DataFrame after its population from Census API data

    Returns:
        (pd.DataFrame): a modified copy of the original DataFrame
    '''
    df["Percent Personal Vehicle for Work Commute"] = (
        df["Work Commute Means: Personal Vehicle"] / 
        df["Work Commute Means: Total Surveyed"]
    )
    df["Percent Carpool for Work Commute"] = (
        df["Work Commute Means: Carpool"] / 
        df["Work Commute Means: Total Surveyed"]
    )
    df["Percent Public Transportation for Work Commute"] = (
        df["Work Commute Means: Public Transportation"] / 
        df["Work Commute Means: Total Surveyed"]
    )
    df["Percent Walks for Work Commute"] = (
        df["Work Commute Means: Walks"] / 
        df["Work Commute Means: Total Surveyed"]
    )
    df["Percent Works from Home/No Commute"] = (
        df["Work Commute Means: Works from Home"] / 
        df["Work Commute Means: Total Surveyed"]
    )

    return df.drop(columns=[
        "Work Commute Means: Total Surveyed",
        "Work Commute Means: Personal Vehicle",
        "Work Commute Means: Carpool",
        "Work Commute Means: Public Transportation",
        "Work Commute Means: Walks",
        "Work Commute Means: Works from Home"
    ])


def _compute_percent_family_type(df):
    '''
    Adds new columns to the DataFrame to capture the percentage of census
    tract householders in which the householder is single or married and
    with or without children. Removes columns used as intermediaries in
    the calculation.
    
    Parameters:
        df (pd.DataFrame): the DataFrame after its population from Census API data

    Returns:
        (pd.DataFrame): a modified copy of the original DataFrame
    '''
    df["Percent Married Couple with Own Children"] = (
        df["Family Type: Married Couple with Own Children"] / 
        df["Family Type: Total Surveyed"]
    )
    df["Percent Married Couple without Own Children"] = (
        df["Family Type: Married Couple without Own Children"] / 
        df["Family Type: Total Surveyed"]
    )
    df["Percent Single Father with Own Children"] = (
        df["Family Type: Single Father with Own Children"] / 
        df["Family Type: Total Surveyed"]
    ) 
    df["Percent Single Mother with Own Children"] = (
        df["Family Type: Single Mother with Own Children"] / 
        df["Family Type: Total Surveyed"]
    )

    return df.drop(columns=[
        "Family Type: Total Surveyed",
        "Family Type: Married Couple with Own Children",
        "Family Type: Married Couple without Own Children",
        "Family Type: Single Father with Own Children",
        "Family Type: Single Mother with Own Children"
    ])


def _compute_percent_geographical_mobility(df):
    '''
    Adds new columns to the DataFrame to capture the geographical mobility of
    census tract residents within the past year. Removes columns used as
    intermediaries in the calculation.

    Parameters:
        df (pd.DataFrame): the DataFrame after its population from Census API data

    Returns:
        (pd.DataFrame): a modified copy of the original DataFrame
    '''
    df["Percent Lived in Same House One Year Ago"] = (
        df["Geographical Mobility: Lived in Same House in Year Ago"] / 
        df["Geographical Mobility: Total Surveyed"]
    )
    df["Percent Moved within Same County Last Year"] = (
        df["Geographical Mobility: Moved within Same County Last Year"] / 
        df["Geographical Mobility: Total Surveyed"]
    )
    df["Percent Moved from Different County in Same State Last Year"] = (
        df["Geographical Mobility: Moved from Different County in Same State Last Year"] / 
        df["Geographical Mobility: Total Surveyed"]
    ) 
    df["Percent Moved from Different State Last Year"] = (
        df["Geographical Mobility: Moved from Different State Last Year"] / 
        df["Geographical Mobility: Total Surveyed"]
    )
    df["Percent Moved from Abroad Last Year"] = (
        df["Geographical Mobility: Moved from Abroad Last Year"] / 
        df["Geographical Mobility: Total Surveyed"]
    )

    return df.drop(columns=[
        "Geographical Mobility: Total Surveyed",
        "Geographical Mobility: Lived in Same House in Year Ago",
        "Geographical Mobility: Moved within Same County Last Year",
        "Geographical Mobility: Moved from Different County in Same State Last Year",
        "Geographical Mobility: Moved from Different State Last Year",
        "Geographical Mobility: Moved from Abroad Last Year"
    ])


def _compute_percent_gross_rent_as_income_share(df):
    '''
    Adds new columns representing gross rent as a share of income
    in the DataFrame. Removes columns used as intermediaries in the calculation.

    Parameters:
        df (pd.DataFrame): the DataFrame after its population from Census API data

    Returns:
        (pd.DataFrame): a modified copy of the original DataFrame
    '''
    df["Percent Gross Rent Less than 10.0 Percent of Income"] = (
        df["Gross Rent to Income: Less than 10.0 Percent"] / 
        df["Gross Rent to Income: Total Surveyed"]
    )
    df["Percent Gross Rent 10.0-14.9 Percent of Income"] = (
        df["Gross Rent to Income: 10.0-14.9 Percent"] / 
        df["Gross Rent to Income: Total Surveyed"]
    )
    df["Percent Gross Rent 15.0-19.9 Percent of Income"] = (
        df["Gross Rent to Income: 15.0-19.9 Percent"] / 
        df["Gross Rent to Income: Total Surveyed"]
    ) 
    df["Percent Gross Rent 20.0-24.9 Percent of Income"] = (
        df["Gross Rent to Income: 20.0-24.9 Percent"] / 
        df["Gross Rent to Income: Total Surveyed"]
    )
    df["Percent Gross Rent 25.0-29.9 Percent of Income"] = (
        df["Gross Rent to Income: 25.0-29.9 Percent"] / 
        df["Gross Rent to Income: Total Surveyed"]
    )
    df["Percent Gross Rent 30.0-34.9 Percent of Income"] = (
        df["Gross Rent to Income: 30.0-34.9 Percent"] / 
        df["Gross Rent to Income: Total Surveyed"]
    )
    df["Percent Gross Rent 35.0-39.9 Percent of Income"] = (
        df["Gross Rent to Income: 35.0-39.9 Percent"] / 
        df["Gross Rent to Income: Total Surveyed"]
    )
    df["Percent Gross Rent 40.0-49.9 Percent of Income"] = (
        df["Gross Rent to Income: 40.0-49.9 Percent"] / 
        df["Gross Rent to Income: Total Surveyed"]
    )
    df["Percent Gross Rent 50.0 Percent or More of Income"] = (
        df["Gross Rent to Income: 50.0 Percent or More"] / 
        df["Gross Rent to Income: Total Surveyed"]
    )

    return df.drop(columns=[
        "Gross Rent to Income: Total Surveyed",
        "Gross Rent to Income: Less than 10.0 Percent",
        "Gross Rent to Income: 10.0-14.9 Percent",
        "Gross Rent to Income: 15.0-19.9 Percent",
        "Gross Rent to Income: 20.0-24.9 Percent",
        "Gross Rent to Income: 25.0-29.9 Percent",
        "Gross Rent to Income: 30.0-34.9 Percent",
        "Gross Rent to Income: 35.0-39.9 Percent",
        "Gross Rent to Income: 40.0-49.9 Percent",
        "Gross Rent to Income: 50.0 Percent or More"
    ])


def _compute_percent_household_size(df):
    '''
    Adds new columns representing percentages of household sizes for owner
    occupied and renter occupied units in the DataFrame.

    Parameters:
        (pd.DataFrame): the DataFrame after its population from Census API data

    Returns:
        (pd.DataFrame): a modified copy of the original DataFrame
    '''
    df["Percent Owner Occupied 1-Person Household"] = (
        df["Household Size: Owner Occupied 1-Person Household"] / 
        df["Household Size: Total Surveyed"]
    )
    df["Percent Owner Occupied 2-Person Household"] = (
        df["Household Size: Owner Occupied 2-Person Household"] / 
        df["Household Size: Total Surveyed"]
    )
    df["Percent Owner Occupied 3-Person Household"] = (
        df["Household Size: Owner Occupied 3-Person Household"] / 
        df["Household Size: Total Surveyed"]
    )
    df["Percent Owner Occupied 4-Person Household"] = (
        df["Household Size: Owner Occupied 4-Person Household"] / 
        df["Household Size: Total Surveyed"]
    )
    df["Percent Owner Occupied 5-Person Household"] = (
        df["Household Size: Owner Occupied 5-Person Household"] / 
        df["Household Size: Total Surveyed"]
    )
    df["Percent Owner Occupied 6-Person Household"] = (
        df["Household Size: Owner Occupied 6-Person Household"] / 
        df["Household Size: Total Surveyed"]
    )
    df["Percent Owner Occupied 7-or-more Person Household"] = (
        df["Household Size: Owner Occupied 7-or-more Person Household"] / 
        df["Household Size: Total Surveyed"]
    )
    df["Percent Renter Occupied 1-Person Household"] = (
        df["Household Size: Renter Occupied 1-Person Household"] / 
        df["Household Size: Total Surveyed"]
    )
    df["Percent Renter Occupied 2-Person Household"] = (
        df["Household Size: Renter Occupied 2-Person Household"] / 
        df["Household Size: Total Surveyed"]
    )
    df["Percent Renter Occupied 3-Person Household"] = (
        df["Household Size: Renter Occupied 3-Person Household"] / 
        df["Household Size: Total Surveyed"]
    )
    df["Percent Renter Occupied 4-Person Household"] = (
        df["Household Size: Renter Occupied 4-Person Household"] / 
        df["Household Size: Total Surveyed"]
    )
    df["Percent Renter Occupied 5-Person Household"] = (
        df["Household Size: Renter Occupied 5-Person Household"] / 
        df["Household Size: Total Surveyed"]
    )
    df["Percent Renter Occupied 6-Person Household"] = (
        df["Household Size: Renter Occupied 6-Person Household"] / 
        df["Household Size: Total Surveyed"]
    )
    df["Percent Renter Occupied 7-or-more Person Household"] = (
        df["Household Size: Renter Occupied 7-or-more Person Household"] / 
        df["Household Size: Total Surveyed"]
    )

    return df.drop(columns=[
        "Household Size: Total Surveyed",
        "Household Size: Owner Occupied 1-Person Household",
        "Household Size: Owner Occupied 2-Person Household",
        "Household Size: Owner Occupied 3-Person Household",
        "Household Size: Owner Occupied 4-Person Household",
        "Household Size: Owner Occupied 5-Person Household",
        "Household Size: Owner Occupied 6-Person Household",
        "Household Size: Owner Occupied 7-or-more Person Household",
        "Household Size: Renter Occupied 1-Person Household",
        "Household Size: Renter Occupied 2-Person Household",
        "Household Size: Renter Occupied 3-Person Household",
        "Household Size: Renter Occupied 4-Person Household",
        "Household Size: Renter Occupied 5-Person Household",
        "Household Size: Renter Occupied 6-Person Household",
        "Household Size: Renter Occupied 7-or-more Person Household"
    ])


def _compute_percent_household_type(df):
    '''
    Adds a new column to the DataFrame to capture the percentage of census
    tract residents who are living alone or classified as a roomer, boarder,
    housemate, roommate, or unmarried partner with respect to the head of
    household. Removes columns used as intermediaries in the calculation.

    Parameters:
        df (pd.DataFrame): the DataFrame after its population from Census API data

    Returns:
        (pd.DataFrame): a modified copy of the original DataFrame
    '''
    df["Percent Women Living Alone"] = (
        df["Household Type: Women Living Alone"] / 
        df["Household Type: Total Surveyed"]
    )
    df["Percent Men Living Alone"] = (
        df["Household Type: Men Living Alone"] / 
        df["Household Type: Total Surveyed"]
    )
    df["Percent Roomers or Boarders"] = (
        df["Household Type: Roomers or Boarders"] / 
        df["Household Type: Total Surveyed"]
    )
    df["Percent Housemates or Roommates"] = (
        df["Household Type: Housemates or Roommates"] / 
        df["Household Type: Total Surveyed"]
    )
    df["Percent Unmarried Partners"] = (
        df["Household Type: Unmarried Partners"] / 
        df["Household Type: Total Surveyed"]
    )

    return df.drop(columns=[
        "Household Type: Total Surveyed",
        "Household Type: Men Living Alone",
        "Household Type: Women Living Alone",
        "Household Type: Roomers or Boarders",
        "Household Type: Housemates or Roommates",
        "Household Type: Unmarried Partners"
    ])


def _compute_percent_income_below_poverty_level(df):
    '''
    Adds a new column to the DataFrame to capture the percentage of census
    tract residents whose income fell below the poverty level in the past 12
    months. Removes columns used as intermediaries in the calculation.

    Parameters:
        df (pd.DataFrame): the DataFrame after its population from Census API data

    Returns:
        (pd.DataFrame): a modified copy of the original DataFrame
    '''
    df["Percent Below Poverty Level in Last 12 Months"] = (
        df["Poverty Status: Below Poverty Level in Last 12 Months"] / 
        df["Poverty Status: Total Surveyed"]
    )

    return df.drop(columns=[
        "Poverty Status: Total Surveyed",
        "Poverty Status: Below Poverty Level in Last 12 Months"
    ])


def _compute_percent_insured(df):
    '''
    Adds two new columns to the DataFrame to capture the percentage of working
    age adults 18-64, both male and female, who have health insurance coverage.
    Removes columnsused as intermediaries in the calculation.

    Parameters:
        df (pd.DataFrame): the DataFrame after its population from Census API data

    Returns:
        (pd.DataFrame): a modified copy of the original DataFrame
    '''
    df["Percent Working Age Men Insured"] = (
        (df["Health Insurance: Male Teens through Mid-Twenties Insured"] +
        df["Health Insurance: Male Mid-Twenties to 34 Insured"] +
        df["Health Insurance: Male 35-44 Insured"] +
        df["Health Insurance: Male 45-54 Insured"] +
        df["Health Insurance: Male 55-64 Insured"]) /
        (df["Health Insurance: Total Male Teens through Mid-Twenties"] + 
        df["Health Insurance: Total Male Mid-Twenties to 34"] +
        df["Health Insurance: Total Male 35-44"] +
        df["Health Insurance: Total Male 45-54"] + 
        df["Health Insurance: Total Male 55-64"])
    )

    df["Percent Working Age Women Insured"] = (
        (df["Health Insurance: Female Teens through Mid-Twenties Insured"] +
        df["Health Insurance: Female Mid-Twenties to 34 Insured"] +
        df["Health Insurance: Female 35-44 Insured"] +
        df["Health Insurance: Female 45-54 Insured"] +
        df["Health Insurance: Female 55-64 Insured"]) /
        (df["Health Insurance: Total Female Teens through Mid-Twenties"] + 
        df["Health Insurance: Total Female Mid-Twenties to 34"] +
        df["Health Insurance: Total Female 35-44"] +
        df["Health Insurance: Total Female 45-54"] + 
        df["Health Insurance: Total Female 55-64"])
    )

    return df.drop(columns=[
        "Health Insurance: Total Surveyed",
        "Health Insurance: Total Male Teens through Mid-Twenties",
        "Health Insurance: Male Teens through Mid-Twenties Insured",
        "Health Insurance: Total Male Mid-Twenties to 34",
        "Health Insurance: Male Mid-Twenties to 34 Insured",
        "Health Insurance: Total Male 35-44",
        "Health Insurance: Male 35-44 Insured",
        "Health Insurance: Total Male 45-54",
        "Health Insurance: Male 45-54 Insured",
        "Health Insurance: Total Male 55-64",
        "Health Insurance: Male 55-64 Insured",
        "Health Insurance: Total Female Teens through Mid-Twenties",
        "Health Insurance: Female Teens through Mid-Twenties Insured",
        "Health Insurance: Total Female Mid-Twenties to 34",
        "Health Insurance: Female Mid-Twenties to 34 Insured",
        "Health Insurance: Total Female 35-44",
        "Health Insurance: Female 35-44 Insured",
        "Health Insurance: Total Female 45-54",
        "Health Insurance: Female 45-54 Insured",
        "Health Insurance: Total Female 55-64",
        "Health Insurance: Female 55-64 Insured"
    ])


def _compute_percent_place_of_birth(df):
    '''
    Adds a new column to the DataFrame to capture the percentage of census
    tract residents who were born in the state in which they are residing.
    Removes columns used as intermediaries in the calculation.

    Parameters:
        df (pd.DataFrame): the DataFrame after its population from Census API data

    Returns:
        (pd.DataFrame): a modified copy of the original DataFrame
    '''
    df["Percent Born in State"] = (
        df["Birthplace: Born in State"] / 
        df["Birthplace: Total Surveyed"]
    )
    df["Percent Foreign Born"] = (
        df["Birthplace: Foreign Born"] / 
        df["Birthplace: Total Surveyed"]
    )

    return df.drop(columns=[
        "Birthplace: Total Surveyed",
        "Birthplace: Born in State",
        "Birthplace: Foreign Born"
    ])


def _compute_percent_races(df):
    '''
    Adds new columns to the DataFrame to capture the percentage of census
    tract residents of different racial backgrounds. Here the "Hispanic or
    Latino" category is treated as a separate race, and all other racial groups
    are non-Hispanic. Removes columns used as intermediaries in the calculation.

    Parameters:
        df (pd.DataFrame): the DataFrame after its population from Census API data

    Returns:
        (pd.DataFrame): a modified copy of the original DataFrame
    '''
    df["Percent White"] = (
        df["Race: White"] / 
        df["Race: Total Surveyed"]
    )
    df["Percent Black"] = (
        df["Race: Black"] / 
        df["Race: Total Surveyed"]
    )
    df["Percent American Indian and Alaska Native"] = (
        df["Race: American Indian and Alaska Native"] / 
        df["Race: Total Surveyed"]
    )
    df["Percent Asian"] = (
        df["Race: Asian"] / 
        df["Race: Total Surveyed"]
    )
    df["Percent Native Hawaiian and Other Pacific Islander"] = (
        df["Race: Native Hawaiian and Other Pacific Islander"] / 
        df["Race: Total Surveyed"]
    )
    df["Percent Other Race"] = (
        df["Race: Other Race"] / 
        df["Race: Total Surveyed"]
    )
    df["Percent Multiracial"] = (
        df["Race: Two or More Races"] / 
        df["Race: Total Surveyed"]
    )
    df["Percent Hispanic"] = (
        df["Race: Hispanic"] / 
        df["Race: Total Surveyed"]
    )

    return df.drop(columns=[
        "Race: White",
        "Race: Black",
        "Race: American Indian and Alaska Native",
        "Race: Asian",
        "Race: Native Hawaiian and Other Pacific Islander",
        "Race: Other Race",
        "Race: Two or More Races",
        "Race: Hispanic"
    ])


def _compute_percent_school_enrollment(df):
    '''
    Adds new columns to the DataFrame to capture school enrollment figures
    for each census tract. Removes columns used as intermediaries in
    the calculation.

    Parameters:
        df (pd.DataFrame): the DataFrame after its population from Census API data

    Returns:
        (pd.DataFrame): a modified copy of the original DataFrame
    '''
    df["Percent in School"] = (
        df["School Enrollment: Total Enrolled in School"] / 
        df["School Enrollment: Total Surveyed"]
    )
    df["Percent in Nursery or Preschool"] = (
        df["School Enrollment: Enrolled in Nursery or Preschool"] / 
        df["School Enrollment: Total Surveyed"]
    )
    df["Percent in K-12"] = (
        (df["School Enrollment: Enrolled in Kindergarten"] + 
            df["School Enrollment: Enrolled in Grades 1-4"] +
            df["School Enrollment: Enrolled in Grades 5-8"] +
            df["School Enrollment: Enrolled in Grades 9-12"]) /
        df["School Enrollment: Total Surveyed"]
    )
    df["Percent in College"] = (
        (df["School Enrollment: Enrolled in Undergrad"] +
            df["School Enrollment: Enrolled in Graduate or Professional School"]) / 
        df["School Enrollment: Total Surveyed"]
    )

    return df.drop(columns=[
        "School Enrollment: Total Surveyed",
        "School Enrollment: Total Enrolled in School",
        "School Enrollment: Enrolled in Nursery or Preschool",
        "School Enrollment: Enrolled in Kindergarten",
        "School Enrollment: Enrolled in Grades 1-4",
        "School Enrollment: Enrolled in Grades 5-8",
        "School Enrollment: Enrolled in Grades 9-12",
        "School Enrollment: Enrolled in Undergrad",
        "School Enrollment: Enrolled in Graduate or Professional School"
    ])


def _compute_percent_single_never_married(df):
    '''
    Adds new columns to the DataFrame to capture the percentage of census
    tract residents of each sex who have never been married. Removes columns
    used as intermediaries in the calculation.

    Parameters:
        df (pd.DataFrame): the DataFrame after its population from Census API data

    Returns:
        (pd.DataFrame): a modified copy of the original DataFrame
    '''
    df["Percent Male Never Married"] = (
        df["Marital Status: Total Male Never Married"] / 
        df["Marital Status: Total Male Surveyed"]
    )
    df["Percent Female Never Married"] = (
        df["Marital Status: Total Female Never Married"] / 
        df["Marital Status: Total Female Surveyed"]
    )

    return df.drop(columns=[
        "Marital Status: Total Surveyed",
        "Marital Status: Total Male Surveyed",
        "Marital Status: Total Male Never Married",
        "Marital Status: Total Female Surveyed",
        "Marital Status: Total Female Never Married"
    ])


def _compute_percent_white_collar(df):
    '''
    Adds a new column to the DataFrame to capture the percentage of census
    tract residents who are "white-collar" workers within the "Management,
    business, science, and arts" occupations. Removes columns used as
    intermediaries in the calculation.

    Parameters:
        df (pd.DataFrame): the DataFrame after its population from Census API data

    Returns:
        (pd.DataFrame): a modified copy of the original DataFrame
    '''
    df["Percent White Collar"] = (
        df["Total White Collar Workers"] / 
        df["Total Occupational Count"]
    )

    return df.drop(columns=[
        "Total Occupational Count",
        "Total White Collar Workers"
    ])


def _parse_config(config_folder_path):
    '''
    Parses a set of configuration files to retrieve the geographies, tables,
    and variables used to build queries against the Census Bureau's Data API.

    Parameters:
        config_folder_path (str): the path to the folder containing the JSON
                                  configuration files

    Returns:
        (ACSDataRequest): the data request
    '''
    # Retrieve ACS data source and cities and years of interest
    with open(f"{config_folder_path}/master.json") as f:
        master_settings = json.load(f)
        acs = master_settings["acs"]
        cities = master_settings["cities"]
        years = master_settings["years"]

    # Retrieve geographies and filter by city name
    with open(f"{config_folder_path}/geos.json") as f:
        geomaps = []
        for g in json.load(f):
            geomapping = jsons.load(g, GeoMapping)
            if geomapping.city.name in cities:
                geomaps.append(geomapping)

    # Retrieve variables for given year and parse into dictionary
    all_var_dicts = {}
    for year in years:
        with open(f"{config_folder_path}/variables/{year}.csv") as f:
            var_dict = {}
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                var_dict[row["Code"]] = row["VariableCustomName"]
            all_var_dicts[year] = var_dict

    return ACSDataRequest(acs, cities, years, geomaps, all_var_dicts)


def _reshape_dataframe(df, variable_dict, year):
    '''
    Reshapes the DataFrame by adding derived feature columns, eliminating
    columns used as intermediaries in the calculations, and resetting the index
    to the GEO_ID.

    Parameters:
        df (pd.DataFrame): the DataFrame after its population from Census API data
        variable_dict (dict<str, str>): a dictionary mapping ACS variable codes
                                        to custom variable names
        year (int): the publication year for the ACS data (e.g., 2013)

    Returns:
        (pd.DataFrame): a modified copy of the original DataFrame
    '''

    # Replace variable code column names with meaningful labels
    updated_df = df.rename(columns=variable_dict[year])

    # Update educational attainment column using custom logic
    compute_by_sex = year == 2010 or year == 2011
    updated_df = _compute_percent_college_graduate(updated_df, compute_by_sex)

    # Finalize remaining columns generated from table variables
    updated_df = _compute_percent_commute_time(updated_df)
    updated_df = _compute_percent_commute_type(updated_df)
    updated_df = _compute_percent_family_type(updated_df)
    updated_df = _compute_percent_geographical_mobility(updated_df)
    updated_df = _compute_percent_gross_rent_as_income_share(updated_df)
    updated_df = _compute_percent_household_size(updated_df)
    updated_df = _compute_percent_household_type(updated_df)
    updated_df = _compute_percent_income_below_poverty_level(updated_df)
    updated_df = _compute_percent_place_of_birth(updated_df)
    updated_df = _compute_percent_races(updated_df)
    updated_df = _compute_percent_school_enrollment(updated_df)
    updated_df = _compute_percent_single_never_married(updated_df)
    updated_df = _compute_percent_white_collar(updated_df)

    # Update index
    updated_df = (updated_df
        .reset_index()
        .drop("index", axis="columns")
        .set_index("GEO_ID")
    )

    # Add "geo11" column to facilitate later spatial joins
    updated_df["geo11"] = updated_df.index.str[-11:]
    updated_df["geo11"] = updated_df["geo11"].astype({'geo11': 'object'})

    return updated_df
  

def _write_output_files(df, acs, year, output_folder_path):
    '''
    Writes the contents of the DataFrame to one or more output CSV files.

    Parameters:
        df (pd.DataFrame): the reshaped DataFrame
        acs (str): the ACS data source (e.g., 'acs5')
        year (int): the publication year for the ACS data (e.g., 2013)
        output_folder_path (str): the path to the folder containing the CSV
                                  output files

    Returns:
        None
    '''
    # Define output columns for CSV file holding socioeconomic data only
    # (To be used for calculating socioeconomic ascent outcome variable)
    socioeconomic_vars = [
        "State",
        "County",
        "Affiliated City",
        "geo11",
        "Year",
        "Median Annual Household Income",
        "Median Monthly Housing Costs",
        "Median Value for Owner Occupied Housing Units",
        "Percent White Collar",
        "Percent College Graduate",
        "Total Population"
    ]

    # Write ACS features to CSV file
    df.to_csv(f"{output_folder_path}/{acs}_{year}_features.csv")

    # Write socioeconomic data to CSV file for later transformation
    df[socioeconomic_vars].to_csv(
        f"{output_folder_path}/{acs}_{year}_socioeconomic_vars.csv"
    )


def orchestrate(config_folder_path, output_folder_path):
    '''
    Orchestrates the retrieval of ACS data from the Census Bureau API
    and saves the output to a set of CSV files.

    Parameters:
        config_folder_path (str): the path to the folder containing the JSON
                                  configuration files
        output_folder_path (str): the path to the folder containing the CSV
                                  output files
    
    Returns:
        (pd.DataFrame): a DataFrame with requested ACS data
    '''
    # Get ACS data request
    acsreq = _parse_config(config_folder_path)

    # Process data for each year
    for year in acsreq.years:

        logging.info(f"Processing ACS data for {year}")

        # Initialize DataFrame
        df = pd.DataFrame()

        # Retrieve variable codes (e.g. "B08303_003E") and add "GEO_ID"
        var_codes = list(acsreq.variables[year].keys())
        var_codes.append("GEO_ID")

        # Retrieve census data from API
        for geomap in acsreq.geomaps:
            logging.info(f"Calling API for {geomap.city.name}")
            temp_df = acsclient.get_census_tracts(
                source=acsreq.acs,
                year=year,
                geomapping=geomap,
                var_codes=var_codes
            )
            df = pd.concat([df, temp_df])
            
        # Reshape DataFrame
        logging.info(f"Reshaping resulting DataFrame")
        df = _reshape_dataframe(df, acsreq.variables, year)

        # Write to output csv
        logging.info(f"Writing data to output CSV files")
        _write_output_files(df, acsreq.acs, year, output_folder_path)

   
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config_folder_path = "acs/config"
    output_folder_path = "acs/outputs"
    orchestrate(config_folder_path, output_folder_path)