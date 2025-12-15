# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.10.6
# ---

# %% [markdown]
# Data from: https://www.ncei.noaa.gov/cdo-web/search?datasetid=GHCND
#
# Collected Daily Summaries from 3 different stations to find average weather across regions of Minnesota: 
# - INTERNATIONAL FALLS INTERNATIONAL AIRPORT, MN US (N Region)
# - MINNEAPOLIS ST. PAUL INTERNATIONAL AIRPORT, MN US (SE Region)
# - ROCHESTER INTERNATIONAL AIRPORT, MN US (S/SE Region)

# %%
import pandas as pd
import numpy as np


# %%
df = pd.read_csv('../data/raw/weather_data.csv')
df.tail(1)

# %% [markdown]
# Weather Type Codes
# - WT01	Fog, ice fog, or freezing fog (or haze): Reduced visibility, increased slickness.
# - WT02	Heavy fog or thick fog: Severely reduced visibility (major crash factor).
# - WT03	Thunder: Often correlated with heavy rain and sudden visibility changes.
# - WT04	Ice pellets, sleet, snow pellets, or small hail: Immediate increase in road slickness and difficulty controlling vehicles.
# - WT05	Hail (larger): Can cause property damage and sudden driver maneuvers.
# - WT06	Glaze or rime (freezing rain): The most dangerous condition for black ice formation.
# - WT08	Smoke or ash: Reduced air quality and significant visibility reduction.
# - WT09	Blowing or drifting snow: Reduced visibility and accumulation creating slick, uneven conditions.

# %% [markdown]
# ## 1. Initial Data 

# %%
df['DATE'] = pd.to_datetime(df['DATE'], format='%m/%d/%Y')
df = df.sort_values(by='DATE', ascending=True)

# Checking NaN values in WT columns - filling NaN with 0
wt_cols = ['WT01', 'WT02', 'WT03', 'WT04', 'WT05', 'WT06', 'WT08', 'WT09']
df[wt_cols] = df[wt_cols].fillna(0)

# Count NaN in TAVG column
cols_to_check = ['TAVG', 'TMAX', 'TMIN', 'PRCP', 'SNOW']
for col in cols_to_check: 
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Initial NaN counts
print(f'Initial NaN values in TAVG: {df["TAVG"].isna().sum()}, NaN values in PRCP: {df["PRCP"].isna().sum()}, NaN values in SNOW: {df["SNOW"].isna().sum()}')

# %% [markdown]
# ## 2. Dealing with NaNs for Numerical Features
#
#

# %%
numerical_features = ['TAVG', 'PRCP', 'SNOW'] 

# First check if TMAX and TMIN are available to find TAVG
df['TAVG_ESTIMATE'] = (df['TMAX'] + df['TMIN']) / 2
df['TAVG'] = df['TAVG'].fillna(df['TAVG_ESTIMATE'])
print(f"NaNs remaining in TAVG after TMAX/TMIN imputation: {df['TAVG'].isnull().sum()}")

# If NaN still present, fill with daily median across stations
for col in numerical_features:
    daily_median_across_stations = df.groupby('DATE')[col].transform('median')
    df[col] = df[col].fillna(daily_median_across_stations)
    print(f"NaNs remaining in {col} after peer median fill: {df[col].isnull().sum()}")

# %% [markdown]
# ## 3. Aggregate and Merge

# %%
daily_median_weather = df.groupby('DATE')[numerical_features].median().reset_index()
daily_median_weather.columns = ['DATE'] + [f'{col}_MEDIAN' for col in numerical_features]

daily_max_wt = df.groupby('DATE')[wt_cols].max().reset_index()
daily_max_wt.columns = ['DATE'] + [f'{col}_MAX' for col in wt_cols]

final_weather_df = pd.merge(daily_median_weather, daily_max_wt, on='DATE', how='inner') 
print(final_weather_df.sample(10))
print(final_weather_df.info())


# %% [markdown]
# ## 4. Output data

# %%
# Create processed data directory if it doesn't exist
import os
processed_dir = '../data/processed'
os.makedirs(processed_dir, exist_ok=True)

# Save the cleaned dataset
cleaned_file_path = os.path.join(processed_dir, 'cleaned_weather.csv')
final_weather_df.to_csv(cleaned_file_path, index=False)

print(f"Cleaned dataset saved to: {cleaned_file_path}")
