'''
Provides access to American Community Survey (ACS) tables and variables
created by the U.S. Census Bureau. 
'''
import censusdata as census
import pandas as pd
import requests


def get_census_estimate_vars(source, year, base_table_code, filters):
    '''
    Directly calls the Census Bureau API endpoint to retrieve the estimate 
    variable codes (i.e., those ending in "E") for the given table.
    Then filters out the estimate variables by their labels.

    Sample API Response Payload:
    {
        "variables": {
            "B11011_007E": {
            "label": "Estimate!!Total!!Family households!!Other family",
            "concept": "HOUSEHOLD TYPE BY UNITS IN STRUCTURE",
            "predicateType": "int",
            "group": "B11011",
            "limit": 0,
            "attributes": "B11011_007EA,B11011_007M,B11011_007MA"
            },
        [...]
    }

    The estimate variable shown above has the code "B11011_007E"; the base part
    of its code before the underscore indicates that it comes from the table 
    "B11011". Its label is "Estimate!!Total!!Family households!!Other family". 
    
    Parameters:
        source (str): the ACS data source (e.g., 'acs5')
        year (int): the publication year for the ACS data (e.g., 2013)
        base_table_code (str): the base code for the ACS table (e.g., "B11011")
        filters (list of str): labels associated with variables to keep
        
    Returns:
        (list of strings): the variable codes
    '''
    r = requests.get(f"https://api.census.gov/data/{year}/acs/{source}/variables.json")
    
    variables = [f"{base_table_code}_001E"]
    
    for k, v in r.json()["variables"].items():  
        if base_table_code in k and k.endswith("E") and v["label"].lower() in filters:
            variables.append(k)
    
    return variables


def get_census_tracts(source, year, geomapping, var_codes):
    '''
    Retrieves all the census tracts for a given GeoMapping.

    Parameters:
        source (str): the ACS data source (e.g., 'acs5')
        year (str): the year of ACS publication
        geomapping (GeoMapping): the geography for which to get ACS data
        var_codes (list of str): the ASC variable codes

    Returns:
        (pd.DataFrame): A DataFrame with the following schema:

        id | geoid | year | state | city | county | variable1 | variable2 | ...
    '''
    df = pd.DataFrame()
    for county in geomapping.counties:
        tracts = census.censusgeo([
            ("state", geomapping.state.fips_code), 
            ("county", county.fips_code), 
            ("tract", "*")])

        downloaded = census.download(source, year, tracts, var_codes)
        downloaded.insert(loc=0, column="state", value=geomapping.state.name)
        downloaded.insert(loc=1, column="county", value=county.name)
        downloaded.insert(loc=2, column="year", value=year)

        df = pd.concat([df, downloaded])

    return df

    