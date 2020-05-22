'''
models.py
'''
from dataclasses import dataclass
from typing import List

@dataclass
class Geo:
    '''
    Represents a generic geography defined by the Census Bureau.
    '''
    name: str
    fips_code: str


@dataclass
class GeoMapping:
    '''
    Represents an association of geographies defined by the Census Bureau--
    specifically a city associated with a state and one or more counties.
    '''
    city: Geo
    state: Geo
    counties: List[Geo]


@dataclass
class ACSVariable:
    '''
    Represents a generic variable in the American Community Survey.
    '''
    full_name: str
    helper_label: str
    code: str
    year: int


@dataclass
class ACSTable:
    '''
    Represents a generic table in the American Community Survey.
    '''
    full_name: str
    helper_label: str
    code: str
    year: int
    filters: List[str]


@dataclass
class ACSDataRequest:
    '''
    Represents the parameters of a data request for the Census Bureau API
    '''
    acs: str
    cities: List[str]
    year: int
    geomaps: List[GeoMapping]
    tables: List[ACSTable]
    variables: List[ACSVariable]