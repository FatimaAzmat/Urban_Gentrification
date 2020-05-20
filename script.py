
import pandas as pd
import censusdata

#relevant columns from ACS data
col_list = ["B02001_001E",'GEO_ID','B01002A_001E','B01002B_001E','B01002C_001E','B01002D_001E','B01002E_001E','B01002F_001E','B01002G_001E', 'B01002I_001E',
 'B01002_001M', 'C02003_003E', 'C02003_004E', 'C02003_005E', 'C02003_006E', 'C02003_007E', 'C02003_008E', 'C02003_009E', 'C02003_010E',
 'C02003_011E', 'C02003_012E', 'C02003_013E', 'C02003_014E', 'C02003_015E', 'C02003_016E', 'C02003_017E', 'C02003_018E', 'C02003_019E',
 'B15003_001E', 'B15003_002E','B15003_003E','B15003_004E','B15003_005E','B15003_006E','B15003_007E','B15003_008E','B15003_009E',
 'B15003_010E','B15003_011E','B15003_012E','B15003_013E','B15003_014E','B15003_015E','B15003_016E','B15003_017E','B15003_018E',
 'B15003_019E','B15003_020E','B15003_021E','B15003_022E','B15003_023E','B15003_024E','B15003_025E']


#create a dataframe of the features, using any year from 2012-2018
acs = censusdata.download("acs5", 2015, censusdata.censusgeo(
    [("state", "17"), ("county", "031"), ("tract", "*")]), col_list)

#extract census tract from the ACS index and separate it into a column and make that the index
census_tract_list = []
for row in acs.index:
    census_tract_list.append(row.geo[2][1])

acs = acs.reset_index()
acs['census_tract'] = pd.Series(census_tract_list)

#create a dictionary mapping ZCTA/zip code to census tract
census_zipcode_relation_filename = 'zcta_tract_rel_10.csv'
census_zipcode_relation = pd.read_csv(census_zipcode_relation_filename, \
    delimiter=',', dtype=str)
acs_tract_to_zipcode = census_zipcode_relation[census_zipcode_relation\
['STATE'] == '17'][['TRACT', 'ZCTA5']]

tract_zipcode = {}
for row in acs_tract_to_zipcode.itertuples():
    tract_zipcode[row[1]] = row[2]

#add zip code as a column in ACS data
acs_to_zipcode = []
for tract in acs['census_tract']:
    if str(tract) in tract_zipcode:
        val = tract_zipcode[str(tract)]
        acs_to_zipcode.append(val)
acs['zipcode'] = pd.Series(acs_to_zipcode)

#Other features
#business licenses
business = pd.read_csv("Business_Licenses.csv")

#potholes reported
potholes_reported = pd.read_csv("311_Service_Requests_-_Pot_Holes_Reported_-_Historical.csv")

#monthly average bus ridership by route
bus_ridership = pd.read_csv("CTA_-_Ridership_-_Bus_Routes_-_Monthly_Day-Type_Averages___Totals.csv")

#eviction data
eviction = pd.read_csv("eviction data.csv")
acs['GEOID'] = acs['GEO_ID'].str[-11:]
eviction['GEOID'] = eviction['GEOID'].apply(str)
acs = pd.merge(acs, eviction, on='GEOID')
acs = acs.drop(columns=['GEO_ID_x', 'GEO_ID_y'])


'''
#data gathering code
#education
c=censusdata.censustable('acs5', 2015, 'B15003')

var_list = []
for k,v in c.items(): 
    if k[-1] == 'E': 
        var_list.append(k) 

#associated var names for education:
for k,v in c.items(): 
    if k[-1] == 'E': 
        name = 'Education:' + ' ' + v['label'] 
            edu_list.append(name)
''' 




