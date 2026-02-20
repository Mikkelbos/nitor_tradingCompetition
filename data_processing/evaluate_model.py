import pandas



import pandas as pd
import numpy as np
import os

def evaluate(data: pd.DataFrame, val_data: pd.DataFrame, model: dict) -> pd.DataFrame:
    """
    Takes the parameter estimates 'model' for the given market,
    looks up the correct market in 'val_data', and multiplies the 
    values with the parameters from the regression. 
    The estimated target is appended to val_data, and a submission 
    file of [id, target] is generated and exported.
    """
    # Defensive copy to avoid SettingWithCopy
    val_data = val_data.copy()
    val_data['target'] = np.nan
    
    for market_name, params in model.items():
        market_mask = val_data['market'] == market_name
        market_df = val_data[market_mask]
        
        if len(market_df) == 0:
            continue
            
        # Initialize predictions with intercept (constant named 'const' by statsmodels)
        preds = pd.Series(params.get('const', 0.0), index=market_df.index)
        
        # Multiply features with their respective coefficients
        for feature, coef in params.items():
            if feature == 'const':
                continue
            
            # Use 0 if the feature is missing from validation data (e.g. dropped dummy categories)
            if feature in market_df.columns:
                # Handle possible NaN values in validation via fillna(0)
                preds += market_df[feature].fillna(0) * coef
            else:
                # If a feature was in train but not in test, handle missing dummy or column
                preds += 0.0 * coef
                
        # Assign generated predictions back to the designated rows
        val_data.loc[market_mask, 'target'] = preds
        
    # Create the submission dataframe with id and target columns
    submit = val_data[['id', 'target']].copy()
    
    # Sort just to be safe and enforce the order of IDs
    submit = submit.sort_values(by='id').reset_index(drop=True)
    
    print("\n--- SUBMISSION SUMMARY ---")
    print(f"Total rows (Shall be 13098):     {len(submit)}")
    print(f"Starts at ID (Shall be 133627):  {submit['id'].min()}")
    print(f"Ends at ID (Shall be 146778):    {submit['id'].max()}")
    print(f"Missing targets (NaNs):          {submit['target'].isna().sum()}")
    print("--------------------------\n")
    
    # Export the submission file back to the root directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    submit_path = os.path.join(BASE_DIR, 'submission.csv')
    submit.to_csv(submit_path, index=False)
    
    print(f"Saved submission to: {submit_path}")
    
    return submit