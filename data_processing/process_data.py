import pandas as pd
import os
import sys
import datetime
from const import *
from estimate_options import *

def add_time_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create dummy variables for each month-hour combination based on delivery_start.
    Format will be UPPERCASEMONTH_Hh, e.g. JANUARY_1h.
    Avoids the dummy variable trap by dropping the first category (first month-hour).
    """
    df = df.copy()
    ds = pd.to_datetime(df["delivery_start"]) 
    month_names = ds.dt.strftime('%B').str.upper()
    hour_strs = ds.dt.hour.astype(str) + 'h'
    month_hour = month_names + '_' + hour_strs
    all_months = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 
                  'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']
    all_hours = [f"{h}h" for h in range(24)]
    all_categories = [f"{m}_{h}" for m in all_months for h in all_hours]
    
    month_hour_cat = pd.Categorical(month_hour, categories=all_categories)

    dummies = pd.get_dummies(month_hour_cat, drop_first=True, dtype=int)
    df = pd.concat([df, dummies], axis=1)
    
    return df


def addTimePrimeDummy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates month dummies (dropping the first to avoid dummy trap)
    and PrimeMorning (hours 7,8,9) / PrimeNight (hours 17,18,19) dummies for each month.
    e.g., JANUARY_PrimeMorning, JUNE_PrimeNight, etc.
    """
    df = df.copy()
    ds = pd.to_datetime(df["delivery_start"])
    
    month_names = ds.dt.strftime('%B').str.upper()
    all_months = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 
                  'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']
    
    month_cat = pd.Categorical(month_names, categories=all_months)
    month_dummies = pd.get_dummies(month_cat, drop_first=True, dtype=int)
    df = pd.concat([df, month_dummies], axis=1)
    
    is_prime_morning = ds.dt.hour.isin([7, 8, 9])
    is_prime_night = ds.dt.hour.isin([17, 18, 19])
    
    for month in all_months:
        df[f"{month}_PrimeMorning"] = ((month_names == month) & is_prime_morning).astype(int)
        df[f"{month}_PrimeNight"] = ((month_names == month) & is_prime_night).astype(int)
        
    return df


def add_weekend_dummy(df:pd.DataFrame) -> pd.DataFrame:
    """
    Create a boolean 'isWeekend' column which is True if the delivery_start
    falls on a Saturday (5) or Sunday (6).
    """
    df = df.copy()
    ds = pd.to_datetime(df["delivery_start"])
    # 5 is Saturday, 6 is Sunday in pandas dt.dayofweek
    df["isWeekend"] = ds.dt.dayofweek.isin([5, 6]).astype(int)
    return df

def add_squared_variable(df: pd.DataFrame, vars: list[str])-> pd.DataFrame:
    """
    Take a list of numeric variables and square them, 
    appending the new features as 'VARNAMEsq' or 'VARNAME_sq'.
    (Using '_sq' format per specification)
    """
    df = df.copy()
    for var in vars:
        if var in df.columns:
            df[f"{var}_sq"] = df[var] ** 2
    return df

def setup():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(BASE_DIR, 'train.csv')
    test_path = os.path.join(BASE_DIR, 'test_for_participants.csv')
    
    df = pd.read_csv(train_path)
    df_noTarget = pd.read_csv(test_path)
    print(f"Loaded data files:\ndf: {df.shape}\ndf_noTarget: {df_noTarget.shape}")

    df = addTimePrimeDummy(df)
    df = add_weekend_dummy(df)
    df_noTarget = addTimePrimeDummy(df_noTarget)
    df_noTarget = add_weekend_dummy(df_noTarget)
    print(f"Added time and weekend dummies, new shape:\ndf: {df.shape}\ndf_noTarget: {df_noTarget.shape}")

    df = add_squared_variable(df, VAR_SQ)
    df_noTarget = add_squared_variable(df_noTarget, VAR_SQ)
    print(f"Added squared variables, new shape:\ndf: {df.shape}\ndf_noTarget: {df_noTarget.shape}")

    
    return df, df_noTarget



