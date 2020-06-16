'''
vishelper.py
@author: Launa Greer
@last updated: June 15, 2020

Facilitates the creation of a data report for a given city.
'''
# General data analysis
import pandas as pd
import numpy as np

# Spatial analysis
import pysal
from splot.esda import moran_scatterplot
from esda import Moran_Local
from scipy.stats import gmean
from splot.esda import lisa_cluster

# Graphing and display
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from IPython.display import display, HTML


# Define variables
AGE_VARS = [
    "Median Age",
    "Female Median Age",
    "Male Median Age",
    "White Median Age",
    "Black Median Age",
    "Asian Median Age",
    "Hispanic or Latino Median Age"
]

RACE_VARS = [
    "Percent White",
    "Percent Black",
    "Percent American Indian and Alaska Native",
    "Percent Asian",
    "Percent Native Hawaiian and Other Pacific Islander",
    "Percent Other Race",
    "Percent Multiracial",
    "Percent Hispanic"
]


def retrieve_city(df_dict, years, city_name):
    '''
    Retrieves data for the given city across all years from the American
    Community Survey (ACS) DataFrame dictionary.
    
    Parameters:
        df_dict (dict<int, pd.DataFrame>): a dictionary with years as keys
                                           and ACS DataFrames as values
        years (list of int): the years for which to retrieve data
        city_name (str): the city name
    
    Returns:
        (pd.DataFrame): ACS data for the given city for all years specified
    '''
    city_df = pd.DataFrame()
    for year in years:
        city_year_df = df_dict[year].query("`Affiliated City` == @city_name")
        city_df = pd.concat([city_df, city_year_df])
        
    return city_df


def clean_city(city_df, drop_tracts=True):
    '''
    Cleans a city DataFrame by replacing the top-coded value "-666666666"
    with NaN and optionally dropping census tracts from across all years
    if in any year the tract has a population of zero.
    
    Parameters:
        city_df (pd.DataFrame): the ACS DataFrame for a given city
        drop_tracts (bool): whether to drop tracts
    
    Returns:
        (pd.DataFrame): the cleaned DataFrame
    '''
    if drop_tracts:
        # Determine which census tracts have a total population of zero
        # in *any* of the years covered
        tracts_to_drop = list(city_df[city_df['Total Population'] == 0]["geo11"].astype(str))

        # Drop tracts from across all years
        print(f'''Dropping the following {len(tracts_to_drop)} unpopulated tracts across all years:\n\n{', '.join(tracts_to_drop)}\n''')

        retain_tract_func = lambda c: str(c) not in tracts_to_drop
        retained_df = city_df[city_df["geo11"].apply(retain_tract_func)]
        original_num_tracts = len(city_df)
        new_num_tracts = len(retained_df)
        city_df = retained_df

        print(f'''{original_num_tracts - new_num_tracts} total rows dropped, leaving {new_num_tracts} left''')

    # Replace top-coded values
    city_df = city_df.replace(-666666666, np.nan)
    
    return city_df


def join_city_with_shapefile(city_df):
    '''
    Joins a DataFrame to a state shapefile.
    
    Parameters:
        (pd.DataFrame): the cleaned ACS DataFrame for a given city
    
    Returns:
        (gpd.GeoDataFrame): the resulting GeoDataFrame
    '''
    # Load state shapefile into GeoDataFrame
    state_name = city_df["State"].values[0].lower()
    shpfile_path = f"../acs/config/shpfiles/{state_name}"
    gdf = gpd.read_file(shpfile_path)
    gdf.rename(columns={"GEOID10": "geo11"}, inplace=True)
    
    # Join state shapefile to census tracts
    city_df = city_df.astype({"geo11":"str"})
    gdf = gdf.astype({"geo11":"str"})
    city_gdf = pd.merge(city_df, gdf, how='inner', on="geo11")
    city_gdf = gpd.GeoDataFrame(city_gdf, crs=gdf.crs)
    
    return city_gdf


def map_net_change(city_gdf, col_name, start_year, end_year, cmap="seismic"):
    '''
    Maps the net change for a given column across all tracts between a
    start year and an end year.
    '''
    # Subset data for each year
    start = city_gdf[city_gdf["Year"] == start_year][["Year", "geo11", "geometry", col_name]]
    end = city_gdf[city_gdf["Year"] == end_year][["Year", "geo11", col_name]]
    
    # Merge years, effectively reshaping DataFrame
    merged = start.merge(end, how="inner", on="geo11", suffixes=(f" {start_year}", f" {end_year}"))
    
    # Compute difference between end and start years and save values in new column
    merged["Change"] = merged[f"{col_name} {end_year}"] - merged[f"{col_name} {start_year}"]
    
    # Plot change
    axes = merged.plot(
        alpha=0.5,
        cmap=cmap,
        column="Change",
        figsize=(15, 25),
        legend=True,
        legend_kwds={
            'label': f"Change in {col_name}",
            'orientation': "horizontal"
        },
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "red",
            "hatch": "///",
            "label": "Missing values"
        })

    axes.set_xlabel("Longitude", {'fontsize': 15, 'fontweight' : 'bold'})
    axes.set_ylabel("Latitude", {'fontsize': 15, 'fontweight' : 'bold'})


def map_time_lapse(city_gdf, col_name, cmap="magma"):
    '''
    Plots a time lapse of a feature column for a given city.
    
    Parameters:
        city_gdf (gpd.GeoDataFrame): a GeoDataFrame containing data for all years
    '''
    
    # Make plots
    fig, axes_grid = plt.subplots(3, 3, sharex='all', sharey='all', figsize=(30,30))
    plt.subplots_adjust(wspace=0.01, right = 0.5, top=0.5)
    axes_dict = {}
    axes_dict[2010] = axes_grid[0][0]
    axes_dict[2011] = axes_grid[0][1]
    axes_dict[2012] = axes_grid[0][2]
    axes_dict[2013] = axes_grid[1][0]
    axes_dict[2014] = axes_grid[1][1]
    axes_dict[2015] = axes_grid[1][2]
    axes_dict[2016] = axes_grid[2][0]
    axes_dict[2017] = axes_grid[2][1]
    axes_dict[2018] = axes_grid[2][2]
    
    # 
    for year in range(2010, 2019):
        subset = city_gdf[city_gdf["Year"] == year][["geometry", col_name]]
        axes_dict[year].set_title(year, fontsize=16, fontweight="bold")
        
        subset.plot(
            ax=axes_dict[year],
            column=col_name,
            cmap=cmap,
            figsize=(25, 25),
            edgecolor="none",
            legend=False,
            legend_kwds={
                'label': col_name,
                'orientation': "horizontal"
            },
            missing_kwds={
                "color": "lightgrey",
                "edgecolor": "red",
                "hatch": "///",
                "label": "Missing values"
            })
        
    # add a big axis, hide frame
    ax_invis = fig.add_subplot(111, frameon=False)
    ax_invis.grid(False)

    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Longitude", fontsize=15, fontweight="bold")
    plt.ylabel("Latitude", fontsize=15, fontweight="bold")
    
    patch_col = axes_dict[2010].collections[0]
    cb = fig.colorbar(patch_col, 
                      ax=axes_grid, 
                      label=col_name, 
                      orientation="vertical")


def map_local_moran(city_gdf, local_moran_dict):
    '''
    Maps Local Moran's statistics against a geography to indicate high-high
    ("HH"), low-low ("LL"), high-low ("HL"), and low-high ("LH") areas of
    spatial correlation.
    '''
    display(HTML(f"<h3>Moran Local Spatial Autocorrelation, 2010-2018</h3>"))
    col_name = "Median Value for Owner Occupied Housing Units"

    # Make plots
    fig, axes_grid = plt.subplots(3, 3, sharex='all', sharey='all', figsize=(30,30))
    plt.subplots_adjust(wspace=0.01, right = 0.5, top=0.5)
    axes_dict = {}
    axes_dict[2010] = axes_grid[0][0]
    axes_dict[2011] = axes_grid[0][1]
    axes_dict[2012] = axes_grid[0][2]
    axes_dict[2013] = axes_grid[1][0]
    axes_dict[2014] = axes_grid[1][1]
    axes_dict[2015] = axes_grid[1][2]
    axes_dict[2016] = axes_grid[2][0]
    axes_dict[2017] = axes_grid[2][1]
    axes_dict[2018] = axes_grid[2][2]
    
    # Make subplots
    for year in range(2010, 2019):
        axes = axes_dict[year]
        axes.set_title(year, fontsize=16, fontweight="bold")
        x_label = axes.get_xaxis().get_label().set_visible(False)
        y_label = axes.get_yaxis().get_label().set_visible(False)

        subset = city_gdf.query("`Year` == @year")[["Year", "geo11", "geometry", "County", col_name]].set_index("geo11")
        lisa_cluster(local_moran_dict[year], subset, p=0.05, figsize=(10,10), ax=axes)

    # Add a big axis, hide frame
    ax_invis = fig.add_subplot(111, frameon=False)
    ax_invis.grid(False)

    # Hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)


def plot_local_moran_time_lapse(city_gdf):
    '''
    Makes a series of scatterplots of Local Moran values from 2010-2018.
    '''
    def compute_gmean(df, tract, weights, column):
        '''
        Computes the geometric mean of the given column.
        '''
        # Compute county medians
        county_median_df = df[["County", column]].groupby("County").median()

        # Extract values from neighboring tracts, if any exist
        neighbor_values = df.loc[weights.neighbors[tract], column].values
        has_neighbors = not all(np.isnan(neighbor_values))

        # Compute and return geometric mean
        if has_neighbors:
            active = neighbor_values[np.logical_not(np.isnan(neighbor_values))]
            geomean = gmean(active)
        else:
            county = df.loc[tract, "County"]
            geomean = county_median_df.loc[county][0]
            
        return geomean
    
    display(HTML(f"<h3>Moran Local Scatterplot of Median Housing Value (2010-18), p=0.05</h3>"))
    local_moran_dict = {}
    
    # Make plots
    col_name = "Median Value for Owner Occupied Housing Units"
    fig, axes_grid = plt.subplots(3, 3, sharex='row', sharey='row', figsize=(30,30))
    plt.subplots_adjust(wspace=0.05, right = 0.5, top=0.45)
    axes_dict = {}
    axes_dict[2010] = axes_grid[0][0]
    axes_dict[2011] = axes_grid[0][1]
    axes_dict[2012] = axes_grid[0][2]
    axes_dict[2013] = axes_grid[1][0]
    axes_dict[2014] = axes_grid[1][1]
    axes_dict[2015] = axes_grid[1][2]
    axes_dict[2016] = axes_grid[2][0]
    axes_dict[2017] = axes_grid[2][1]
    axes_dict[2018] = axes_grid[2][2]
    
    # For each year from 2010 through 2018
    for year in range(2010, 2019):
        # Subset data
        subset = city_gdf.query("`Year` == @year")[["Year", "geo11", "geometry", "County", col_name]].set_index("geo11")
        
        # Set up weights
        weights = pysal.lib.weights.Queen.from_dataframe(subset.reset_index(), idVariable="geo11")
        weights.transform = 'r'

        # Fill nans with geometric mean if possible and column median otherwise
        incomplete_tracts = subset[subset[col_name].isna()].index.values
        for tract in incomplete_tracts:
            subset.loc[tract, col_name] = compute_gmean(subset, tract, weights, col_name)
            
        local_morans = Moran_Local(subset[col_name].values, weights)
        moran_scatterplot(local_morans, p=0.05, ax=axes_dict[year])
        
        x_label = axes_dict[year].get_xaxis().get_label().set_visible(False)
        y_label = axes_dict[year].get_yaxis().get_label().set_visible(False)
        axes_dict[year].set_title(year, fontsize=16, fontweight="bold")
        
        local_moran_dict[year] = local_morans
        
    # add a big axis, hide frame
    ax_invis = fig.add_subplot(111, frameon=False)
    ax_invis.grid(False)

    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Median Housing Value", fontsize=15, fontweight="bold")
    plt.ylabel("Spatial Lag of Median Housing Value", fontsize=15, fontweight="bold")
    
    #plt.suptitle("Chicago | Moran Local Scatterplot of Median Housing Value (2010-18)", ha="center")
    plt.savefig("scatterplot_moran.png", dpi="figure", bbox_inches='tight', pad_inches = 0)
    
    return local_moran_dict


def plot_histogram(df, col_name, bins, title, xlabel, ylabel):
    '''
    Creates a histogram of a continuous variable in the DataFrame.

    Parameters:
        df (pd.DataFrame): the DataFrame
        col_name (str): the name of the column to plot
        bins (int): the number of bins to use
        title (str): the histogram title
        xlabel (str): the label for the histogram's x-axis
        ylabel (str): the label for the histogram's y-axis

    Returns:
        None
    '''
    title_font_dict = {'fontsize': 20, 'fontweight' : 'bold'}
    axes_font_dict = {'fontsize': 18, 'fontweight' : 'bold'}
    
    axes = sns.distplot(df[col_name], bins=bins)
    axes.set_title(title, title_font_dict)
    axes.set_xlabel(xlabel, axes_font_dict)
    axes.set_ylabel(ylabel, axes_font_dict)
    
    return axes


def plot_regplot_time_lapse(city_gdf, col_name_x, col_name_y):
    '''
    Makes a regression plot of two variables from 2010-2018.
    '''
    display(HTML(f"<h3>{col_name_x} vs. {col_name_y}, 2010-2018</h3>"))
    
    # Make plots
    fig, axes_grid = plt.subplots(3, 3, sharex='all', sharey='all', figsize=(30,30))
    plt.subplots_adjust(wspace=0.01, right = 0.5, top=0.5)
    axes_dict = {}
    axes_dict[2010] = axes_grid[0][0]
    axes_dict[2011] = axes_grid[0][1]
    axes_dict[2012] = axes_grid[0][2]
    axes_dict[2013] = axes_grid[1][0]
    axes_dict[2014] = axes_grid[1][1]
    axes_dict[2015] = axes_grid[1][2]
    axes_dict[2016] = axes_grid[2][0]
    axes_dict[2017] = axes_grid[2][1]
    axes_dict[2018] = axes_grid[2][2]
    
    # 
    for year in range(2010, 2019):
        subset = city_gdf[city_gdf["Year"] == year][["geometry", col_name_x, col_name_y]]
        axes_dict[year].set_title(year, fontsize=16, fontweight="bold")
        
        sns.regplot(
            ax=axes_dict[year],
            x=col_name_x,
            y=col_name_y,
            data=subset,
            line_kws={'color':'red'})
        
        x_label = axes_dict[year].get_xaxis().get_label().set_visible(False)
        y_label = axes_dict[year].get_yaxis().get_label().set_visible(False)

    # add a big axis, hide frame
    ax_invis = fig.add_subplot(111, frameon=False)
    ax_invis.grid(False)

    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel(f"{col_name_x} ($1k)", fontsize=15, fontweight="bold")
    plt.ylabel(f"{col_name_y} ($1k)", fontsize=15, fontweight="bold")


def create_city_geoframe(df_dict, years, city_name):
    '''
    Creates a city GeoDataFrame.
    '''
    city_df = retrieve_city(df_dict, years, city_name)
    city_df = clean_city(city_df)
    city_gdf = join_city_with_shapefile(city_df)

    return city_gdf


def display_statistics(city_gdf, col_names):
    '''
    Displays basic descriptive statistic for a column in a city GeoDataFrame.
    '''
    subset = city_gdf[col_names]
    display(subset.describe().transpose())


def run_report(city_gdf):
    '''
    Generates a data report for a given city.
    '''
    # AGE
    display(HTML("<h2>Age</h2>"))
    display(HTML("<h3>Statistics for All Tracts and Years</h3>"))
    display_statistics(city_gdf, AGE_VARS)

    display(HTML("<h3>Time Lapse of Median Age, 2010-2018</h3>"))
    map_time_lapse(city_gdf, "Median Age", cmap="magma")
    plt.pause(2)

    display(HTML("<h3>Net Change in Median Age, 2010-2018</h3>"))
    map_net_change(city_gdf, "Median Age", 2010, 2018, cmap="seismic")
    plt.pause(2)

    # RACE
    display(HTML("<h2>Race</h2>"))
    display(HTML("<h3>Statistics for All Tracts and Years</h3>"))
    display_statistics(city_gdf, RACE_VARS)

    display(HTML("<h3>Time Lapse of Median Age, 2010-2018</h3>"))
    map_time_lapse(city_gdf, "Median Age", cmap="magma")
    plt.pause(2)

    display(HTML("<h3>Net Change in Percent White, 2010-2018</h3>"))
    map_net_change(city_gdf, "Percent White", 2010, 2018, cmap="seismic")
    plt.pause(2)

    # MEDIAN ANNUAL HOUSEHOLD INCOME
    display(HTML("<h2>Median Annual Household Income</h2>"))
    display(HTML("<h3>Statistics for All Tracts and Years</h3>"))
    display_statistics(city_gdf, "Median Annual Household Income")
    sns.boxplot(x=city_gdf["Median Annual Household Income"])

    display(HTML("<h3>Time Lapse of Median Annual Household Income, 2010-2018</h3>"))
    map_time_lapse(city_gdf, "Median Annual Household Income", cmap="magma")
    plt.pause(2)

    display(HTML("<h3>Net Change in Median Annual Household Income, 2010-2018</h3>"))
    map_net_change(city_gdf, "Median Annual Household Income", 2010, 2018, cmap="seismic")
    plt.pause(2)

    # MEDIAN MONTHLY HOUSING COSTS
    display(HTML("<h2>Median Monthly Housing Costs</h2>"))
    display(HTML("<h3>Statistics for All Tracts and Years</h3>"))
    display_statistics(city_gdf, "Median Monthly Housing Costs")
    sns.boxplot(x=city_gdf["Median Monthly Housing Costs"])

    display(HTML("<h3>Time Lapse of Median Monthly Housing Costs, 2010-2018</h3>"))
    map_time_lapse(city_gdf, "Median Monthly Housing Costs", cmap="magma")
    plt.pause(2)

    display(HTML("<h3>Net Change in Median Monthly Housing Costs, 2010-2018</h3>"))
    map_net_change(city_gdf, "Median Monthly Housing Costs", 2010, 2018, cmap="seismic")
    plt.pause(2)

    # PERCENT WHITE COLLAR
    display(HTML("<h2>Percent White Collar</h2>"))
    display(HTML("<h3>Statistics for All Tracts and Years</h3>"))
    display_statistics(city_gdf, "Percent White Collar")
    sns.boxplot(x=city_gdf["Percent White Collar"])

    display(HTML("<h3>Time Lapse of Percent White Collar, 2010-2018</h3>"))
    map_time_lapse(city_gdf, "Percent White Collar", cmap="magma")
    plt.pause(2)

    display(HTML("<h3>Net Change in Percent White Collar, 2010-2018</h3>"))
    map_net_change(city_gdf, "Percent White Collar", 2010, 2018, cmap="seismic")
    plt.pause(2)

    # PERCENT COLLEGE GRADUATE
    display(HTML("<h2>Percent College Graduate</h2>"))
    display(HTML("<h3>Statistics for All Tracts and Years</h3>"))
    display_statistics(city_gdf, "Percent College Graduate")
    sns.boxplot(x=city_gdf["Percent College Graduate"])

    display(HTML("<h3>Time Lapse of Percent College Graduate, 2010-2018</h3>"))
    map_time_lapse(city_gdf, "Percent College Graduate", cmap="magma")
    plt.pause(2)

    display(HTML("<h3>Net Change in Percent College Graduate, 2010-2018</h3>"))
    map_net_change(city_gdf, "Percent College Graduate", 2010, 2018, cmap="seismic")
    plt.pause(2)

    # MEDIAN VALUE FOR OWNER-OCCUPIED HOUSING UNITS
    display(HTML("<h2>Median Value for Owner Occupied Housing Units</h2>"))
    display(HTML("<h3>Statistics for All Tracts and Years</h3>"))
    display_statistics(city_gdf, "Median Value for Owner Occupied Housing Units")
    sns.boxplot(x=city_gdf["Median Value for Owner Occupied Housing Units"])

    display(HTML("<h3>Time Lapse of Median Value for Owner Occupied Housing Units, 2010-2018</h3>"))
    map_time_lapse(city_gdf, "Median Value for Owner Occupied Housing Units", cmap="magma")
    plt.pause(2)

    display(HTML("<h3>Net Change in Median Value for Owner Occupied Housing Units, 2010-2018</h3>"))
    map_net_change(city_gdf, "Median Value for Owner Occupied Housing Units", 2010, 2018, cmap="seismic")
    plt.pause(2)

    display(HTML("<h3>Median Home Value Spatial Lag, 2010-2018</h3>"))
    city_gdf["Median Home Value"] = city_gdf["Median Value for Owner Occupied Housing Units"] / 1000
    city_gdf["Median Home Value Spatial Lag"] = city_gdf["Median House Value Spatial Lag"] / 1000
    plot_regplot_time_lapse(city_gdf, "Median Home Value", "Median Home Value Spatial Lag")
    plt.pause(2)

    local_moran_dict = plot_local_moran_time_lapse(city_gdf)
    plt.pause(2)

    map_local_moran(city_gdf, local_moran_dict)
