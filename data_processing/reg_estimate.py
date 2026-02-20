import pandas as pd
import statsmodels.api as sm
from estimate_options import *

def run_regression(data: pd.DataFrame) -> dict:
    grouped_data = data.groupby("market")
    market_params = {}
    
    for market_name, group in grouped_data:
        # Only drop columns that actually exist in the dataframe
        cols_to_drop = [c for c in DROP_COLS if c in group.columns]
        
        # Prepare X and y
        X = group.drop(columns=cols_to_drop)
        # Drop rows with NaN to allow regression to work
        valid_idx = X.dropna().index
        if "target" in group.columns:
            valid_idx = valid_idx.intersection(group["target"].dropna().index)
            
            if len(valid_idx) == 0:
                print(f"Skipping market {market_name} due to lack of valid data.")
                continue
                
            X_clean = X.loc[valid_idx]
            y_clean = group.loc[valid_idx, "target"].astype(float)
            
            # Using statsmodels to get a detailed statistical summary
            # We explicitly add a constant (intercept) to the variables
            X_clean_with_const = sm.add_constant(X_clean.astype(float))
            
            # Fit model
            model = sm.OLS(y_clean, X_clean_with_const).fit()
            
            # Print the regression summary to the console so you can inspect significance
            print(f"\n{'='*20} MARKET: {market_name} {'='*20}")
            print(model.summary())
            
            # Construct parameter dictionary from the statsmodels parameters
            # model.params is a pandas Series indexed by the variable names
            market_params[market_name] = model.params.to_dict()
        else:
            print(f"Error: 'target' not found in data for market {market_name}")
            
    return market_params

