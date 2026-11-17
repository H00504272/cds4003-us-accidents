"""Data preparation from the CLO2 report, as one function.

prepare() runs the same steps as the CLO2 notebook and returns two tables
with the same 497,097 rows:
  clean    - the CLO2 result, with missing values filled on the whole file
  unfilled - the same rows and columns, but the missing values are left empty,
             so the modelling notebook can fill them using the training data only
"""
import numpy as np
import pandas as pd

LIMITS = {'Temperature(F)': (-60, 135), 'Wind_Speed(mph)': (0, 200),
          'Pressure(in)': (20, 33), 'Visibility(mi)': (0, 50)}
DROP = ['ID', 'Source', 'Description', 'Street', 'Zipcode', 'Airport_Code', 'Weather_Timestamp',
        'Country', 'Turning_Loop', 'Roundabout', 'End_Time', 'End_Lat', 'End_Lng', 'Distance(mi)',
        'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight', 'Wind_Chill(F)']
POI = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
       'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal']
WIND_MAP = {'CALM': 'Calm', 'VAR': 'Variable', 'N': 'North', 'S': 'South', 'E': 'East', 'W': 'West'}
GROUPS = [('Clear', ('fair', 'clear')),
          ('Cloudy', ('cloud', 'overcast')),
          ('Low Visibility', ('fog', 'mist', 'haze', 'smoke', 'dust', 'sand')),
          ('Rain', ('rain', 'drizzle', 'shower')),
          ('Snow/Ice', ('snow', 'sleet', 'wintry', 'ice', 'hail', 'freezing')),
          ('Thunderstorm', ('thunder', 't-storm', 'tornado'))]


def prepare(path):
    df = pd.read_csv(path, low_memory=False)
    # Step 1: build the regression target
    df['Start_Time'] = pd.to_datetime(df['Start_Time'].str[:19])
    df['End_Time'] = pd.to_datetime(df['End_Time'].str[:19])
    df['Duration_Minutes'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
    # Step 2: keep 5 minutes to 24 hours
    df = df[(df['Duration_Minutes'] >= 5) & (df['Duration_Minutes'] <= 1440)].copy()
    # Step 3: drop 18 attributes
    df = df.drop(columns=DROP)
    # Step 4: impossible weather values -> missing
    for c, (lo, hi) in LIMITS.items():
        df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan
    # Step 5: one label per wind direction
    df['Wind_Direction'] = df['Wind_Direction'].replace(WIND_MAP)
    # Step 6: new attributes
    df['Hour'] = df['Start_Time'].dt.hour
    df['Month'] = df['Start_Time'].dt.month
    df['Year'] = df['Start_Time'].dt.year
    df['DayOfWeek'] = df['Start_Time'].dt.day_name()
    df['Is_Weekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday']).astype(int)
    df['Is_RushHour'] = df['Hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)
    df['Season'] = np.select([df['Month'].isin([12, 1, 2]), df['Month'] <= 5, df['Month'] <= 8],
                             ['Winter', 'Spring', 'Summer'], default='Autumn')
    df['POI_Count'] = df[POI].astype(int).sum(axis=1)
    cond = df['Weather_Condition'].str.lower().fillna('')
    wg = pd.Series('Other', index=df.index)
    for name, words in GROUPS:
        wg[cond.apply(lambda s: any(w in s for w in words))] = name
    wg[df['Weather_Condition'].isna()] = np.nan
    df['Weather_Group'] = wg
    # Step 8 (done here so both tables have the same columns)
    unfilled = df.drop(columns=['Weather_Condition', 'Start_Time'])
    # Step 7: fill on the whole file (CLO2 version)
    clean = unfilled.copy()
    clean['Precipitation(in)'] = clean['Precipitation(in)'].fillna(0)
    for c in clean.columns:
        if clean[c].isna().any():
            v = clean[c].mean() if pd.api.types.is_numeric_dtype(clean[c]) else clean[c].mode().iloc[0]
            clean[c] = clean[c].fillna(v)
    # Step 9: remove exact duplicates (found on the filled table, as in CLO2)
    keep = ~clean.duplicated()
    return clean[keep], unfilled[keep]
