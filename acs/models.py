'''
models.py

Representations of ACS geographies and data requests.
'''

from dataclasses import dataclass
from typing import List, Dict

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
class ACSDataRequest:
    '''
    Represents the parameters of a data request for the Census Bureau API.
    '''
    acs: str
    cities: List[str]
    year: int
    geomaps: List[GeoMapping]
    variables: Dict[str, str]