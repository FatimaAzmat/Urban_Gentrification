'''
dataretrieval.py
'''

import acsclient
import censusdata as census
import json
import jsons
import pandas as pd

from models import GeoMapping, ACSVariable, ACSTable, ACSDataRequest
from typing import List


def _compute_percent_college_graduate(df):
    '''
    Adds a new column to the DataFrame to capture the percentage of census
    tract residents who hold an associate's degree or higher. Assumes that the
    columns "B15003_001E", "B15003_021E", "B15003_022E", "B15003_023E",
    "B15003_024E", and "B15003_025E" exist in the DataFrame as integer types.

    Parameters:
        (pd.DataFrame): the DataFrame after its population from Census API data

    Returns:
        (pd.DataFrame): a modified copy of the original DataFrame
    '''
    educ_var_name_dict = {
        "B15003_001E": "Total Educational Attainment Count",
        "B15003_021E": "Number Associates",
        "B15003_022E": "Number Bachelors",
        "B15003_023E": "Number Masters",
        "B15003_024E": "Number Professional School",
        "B15003_025E": "Number Doctorates"
    }

    updated_df = df.rename(columns=educ_var_name_dict)

    updated_df["Percent College Graduate"] = (
        (updated_df["Number Associates"] +
        updated_df["Number Bachelors"] +
        updated_df["Number Masters"] +
        updated_df["Number Professional School"] +
        updated_df["Number Doctorates"]) /
        updated_df["Total Educational Attainment Count"]     
    )

    return updated_df.drop(columns=list(educ_var_name_dict.values()))


def _compute_percent_white_collar(df):
    '''
    Adds a new column to the DataFrame to capture the percentage of census
    tract residents who are "white-collar" workers within the "Management,
    business, science, and arts" occupations. Assumes that the columns
    "C24060_001E" and "C24060_002E" exist in the DataFrame as integer types.

    Parameters:
        (pd.DataFrame): the DataFrame after its population from Census API data

    Returns:
        (pd.DataFrame): a modified copy of the original DataFrame
    '''
    occup_var_name_dict = {
        "C24060_001E": "Total Occupational Count",
        "C24060_002E": "Total White Collar Workers"       
    }
    updated_df = df.rename(columns=occup_var_name_dict)

    updated_df["Percent White Collar"] = (
        updated_df["Total White Collar Workers"] / 
        updated_df["Total Occupational Count"]
    )

    return updated_df.drop(columns=list(occup_var_name_dict.values()))


def _get_table_variables(acs, year, tables):
    '''
    Filters a list of ACSTables to include only those pertaining to the given
    year and then calls the Census Bureau API endpoint to receive all of the
    specified estimate variables for those tables.

    Parameters:
        acs (str): the ACS data source (e.g., 'acs5')
        year (int): the publication year for the ACS data (e.g., 2013)
        tables (list of ACSTable): the tables for which to retrieve data

    Returns:
        (list of str): the table variable codes
    '''
    var_codes = []
    for table in tables:
        if table.year == year:
            table_var_codes = acsclient.get_census_estimate_vars(
                source=acs,
                year=year,
                base_table_code=table.code,
                filters=table.filters
            )
            var_codes.extend(table_var_codes)
    
    return var_codes


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
        year = master_settings["year"]

    # Retrieve geographies and filter by city name
    with open(f"{config_folder_path}/geos.json") as f:
        geomaps = []
        for g in json.load(f):
            geomapping = jsons.load(g, GeoMapping)
            if geomapping.city.name in cities:
                geomaps.append(geomapping)

    # Retrieve variable tables and filter by year
    with open(f"{config_folder_path}/tables.json") as f:
        tables = []
        for t in json.load(f):
            table = jsons.load(t, ACSTable)
            if table.year == year:
                tables.append(table)

    # Retrieve variables and filter by year
    with open(f"{config_folder_path}/variables.json") as f:
        variables = []
        for v in json.load(f):
            var = jsons.load(v, ACSVariable)
            if var.year == year:
                variables.append(var)

    return ACSDataRequest(acs, cities, year, geomaps, tables, variables)


def _reshape_dataframe(df, asc_var_dict):
    '''
    Reshapes the DataFrame by: (1) replacing the educational attainment columns
    with a "Percent College Graduate" column, (2) replacing the occupational
    columns with a "Percent White Collar" column, (3) renaming the other ACS
    variable columns with their "user-friendly" names ("helper labels"), and
    (4) resetting the index to the GEO_ID

    Parameters:
        (pd.DataFrame): the DataFrame after its population from Census API data

    Returns:
        (pd.DataFrame): a modified copy of the original DataFrame
    '''
    # Computer percent white-collar and college educated
    updated_df = _compute_percent_white_collar(df)
    updated_df = _compute_percent_college_graduate(updated_df)

    # Rename remaining columns and correct index
    updated_df = (updated_df
        .rename(columns=asc_var_dict)
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
    df.to_csv(f"{output_folder_path}/{acs}_{year}_socioeconomic_vars.csv")


def orchestrate(config_folder_path, output_folder_path):
    '''
    Orchestrates the retrieval of ACS data from the Census Bureau API.

    Parameters:
        config_folder_path (str): the path to the folder containing the JSON
                                  configuration files
        output_folder_path (str): the path to the folder containing the CSV
                                  output files
    
    Returns:
        (pd.DataFrame): a DataFrame with requested ACS data
    '''
    # Get ACS request
    acsreq = _parse_config(config_folder_path)

    # Initialize DataFrame
    df = pd.DataFrame()

    # Pull variable codes from tables
    var_codes = _get_table_variables(acsreq.acs, acsreq.year, acsreq.tables)

    # Get remaining variable codes parsed directly from config file
    var_codes.extend([v.code for v in acsreq.variables if v.year == acsreq.year])

    # Add geo id to variable codes by default
    var_codes.append("GEO_ID")

    # Retrieve census data from API
    for geomap in acsreq.geomaps:
        temp_df = acsclient.get_census_tracts(
            source=acsreq.acs,
            year=acsreq.year,
            geomapping=geomap,
            var_codes=var_codes
        )
        df = pd.concat([df, temp_df])
        
    # Reshape DataFrame
    acs_var_dict = {var.code: var.helper_label for var in acsreq.variables}
    df = _reshape_dataframe(df, acs_var_dict)

    # Write to output csv
    _write_output_files(df, acsreq.acs, acsreq.year, output_folder_path)

    
if __name__ == "__main__":
    config_folder_path = "acs/config"
    output_folder_path = "acs/outputs"
    orchestrate(config_folder_path, output_folder_path)