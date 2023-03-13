import pandas as pd
import dask.dataframe as dd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

# The Temporal Class contains methods to extract some basic temporal information from user_logs
# This is separate from the feature_engineering packages, as this must occur PRIOR to merging
# In effect this is a feature engineering prior to preprocessing

class UserLogsFeatures:
    
    # Run the methods
    # Use temporary variables and return them, for memory's sake
    def user_logs_features(self):
        
        logs = self.__group_logs__()

        # For each variable, get the slope and the mean and merge that on the aggregated dataframe
        for var in ['num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'total_secs']:
            print("Calculating Temporal for: ", var)
            
            slopes = self.__get_slopes__(logs, var)
            means  = self.__get_means__(logs, var)
            
            # If first run, define it, else append
            if var == 'num_25': 
                results = means.merge(slopes, on='msno', how='left')
            else:
                results = results.merge(means, on='msno', how='left').merge(slopes, on='msno', how='left')
  
        activity = pd.DataFrame(logs.apply(
            lambda x: x['date'].count() / (x['date'].max() - x['date'].min()) if x['date'].count() > 1 else 1 
        )).rename(columns={0:'activity'}).clip(upper=1.0)
       
        # Add in an activity tracker
        results = results.merge(activity, on='msno', how='left') 
        #results = activity

        return results
        
        
    # Get slope from dataframe, and return the dataframe of it
    def __get_slopes__(self, grouped, variable):
        return pd.DataFrame(grouped.apply(lambda x: linregress(x.date, x[variable])[0])).rename(columns={0:variable+'_slope'})
    
    # Get Mean from dataframe
    def __get_means__(self, grouped, variable):
        return pd.DataFrame(grouped.apply(lambda x: x[variable].mean())).rename(columns={0:variable+'_mean'})
                                                  
    # Need to group by msno and apply on the grouping
    def __group_logs__(self):
        return self.user_logs_df.groupby('msno')
        
