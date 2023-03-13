import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from src import dataset as ds


# Preprocessing class to encapsulate all the cleaning/feature generation
# Class takes in the data, cleans it, then outputs the cleaned version
# Inherited by Dataset, grouped here for clarity
class Preprocessor:
        
        
    def preprocess(self):
        self.process_members()
        self.process_transactions()
        self.process_user_logs()

    #### AUXILIARY METHODS ####
        
    # Loaded Fill method for convenience    
    def __fillna__(self, this_data, column, value):
        this_data[column] = this_data[column].fillna(value)
        
    # Loaded Replacer method for convenience
    def __replace__(self, this_data, column, replacer, value):
        this_data[column] = this_data[column].replace(to_replace=replacer, value=value)
        
    # Loaded outlier handler
    def __handler__(self, this_data, column, limit, median):
        this_data[column] = this_data[column].apply(
            lambda x: x if ((x <= limit) and (x >= -limit)) else np.nan
        )
        self.__fillna__(this_data,column,median)
        
        
        
    #### MAIN METHODS ####
        
    # Process Members    
    # NAs will be sent to zero when appropriate
    def process_members(self):

        # Relabel Genders (0 for NA, 1 for male, 2 for female)
        self.__fillna__(self.members_df,'gender',0)
        self.__replace__(self.members_df,'gender','male',1)
        self.__replace__(self.members_df,'gender','female',2)
               
        # Let Cities, when NA, be zero (rest are 1-22)
        self.__fillna__(self.members_df,'city',0)
        
        # Set Registered Via mode to 0 when NA
        self.__fillna__(self.members_df,'registered_via',0)
        
        # Set outliers to NA, then fill the NA to the mean of the remaining data, which is 29
        self.members_df['bd'] = self.members_df['bd'].apply(
            lambda x: x if (x < 70.0) and (x > 0.0) else np.nan
        )
        self.__fillna__(self.members_df,'bd',29)
        
        # Registration Init Time is labelled via integer - fill NA with mean date, then convert to datetime
        self.__fillna__(self.members_df,'registration_init_time',20130512.0)
        #self.members_df['registration_init_time'] = pd.to_datetime(self.members_df['registration_init_time'], format='%Y%m%d')

        
    # Process Transactions
    # NAs sent to 0 or the mean/median as appropriate
    def process_transactions(self):
        
        # Replace payment method to 0
        self.__fillna__(self.transactions_df,'payment_method_id',0)
        
        # Payment Plan Days - Remove anything more than a month (since we only care about classifying a month ahead)
        # Replacing with 30.0, since anything out of range is the "end of month"
        self.transactions_df['payment_plan_days'] = self.transactions_df['payment_plan_days'].apply(
            lambda x: x if (x <= 30.0) else np.nan
        )
        self.__fillna__(self.transactions_df,'payment_plan_days',30.0)
        
        # List Price, Removing outliers (from BoxPlot, IQR around 210), replace with median when NAN
        self.transactions_df['plan_list_price'] = self.transactions_df['plan_list_price'].apply(
            lambda x: x if (x <= 200.0) else np.nan
        )
        self.__fillna__(self.transactions_df,'plan_list_price',149.0)

        # Actual payment, same treatment
        self.transactions_df['actual_amount_paid'] = self.transactions_df['actual_amount_paid'].apply(
            lambda x: x if (x <= 200.0) else np.nan
        )
        self.__fillna__(self.transactions_df,'actual_amount_paid',149.0)

        # Treat Auto Renew like others, 0=NAN, 1=No, 2=Yes
        # Order here is important, as we want to replace, then fill NA (so 0s dont overlap)
        self.__replace__(self.transactions_df,'is_auto_renew',1,2)
        self.__replace__(self.transactions_df,'is_auto_renew',0,1)
        self.__fillna__(self.transactions_df,'is_auto_renew',0)
        
        # Transaction Date, treat like members dates (Here use median, since the mean lies in open data)
        self.__fillna__(self.transactions_df,'transaction_date',20170315.0)
        #self.transactions_df['transaction_date'] = pd.to_datetime(self.transactions_df['transaction_date'], format='%Y%m%d')
        
        # Membership Expiry Date, treat same as other dates (Median here as well, though its similar to mean)
        self.__fillna__(self.transactions_df,'membership_expire_date',20170419.0)
        #self.transactions_df['membership_expire_date'] = pd.to_datetime(self.transactions_df['membership_expire_date'], format='%Y%m%d')

        # Cancelling status, treat like the other booleans
        self.__replace__(self.transactions_df,'is_cancel',1,2)
        self.__replace__(self.transactions_df,'is_cancel',0,1)
        self.__fillna__(self.transactions_df,'is_cancel',0)
        

        
    # Process User Logs
    # NAs sent to 0 or the mean as appropriate
    def process_user_logs(self):
        
        # Date - handle like above
        self.__fillna__(self.user_logs_df,'date',20170316.0)
        # Keep a string date just in case
        #self.user_logs_df['raw_date'] = self.user_logs_df['date']
        #self.user_logs_df['date'] = pd.to_datetime(self.user_logs_df['date'], format='%Y%m%d')
        
        # N25 - Handle Outliers then Fill w/Median
        self.user_logs_df['num_25'] = self.user_logs_df['num_25'].apply(
            lambda x: x if (x <= 17.0) else np.nan
        )
        self.__fillna__(self.user_logs_df,'num_25',2.0)
        
        # N50 - Handle Outliers then Fill w/Median
        self.user_logs_df['num_50'] = self.user_logs_df['num_50'].apply(
            lambda x: x if (x <= 5.0) else np.nan
        )
        self.__fillna__(self.user_logs_df,'num_50',0.0)
        
        # N75 - Handle Outliers then Fill w/Median
        self.user_logs_df['num_75'] = self.user_logs_df['num_75'].apply(
            lambda x: x if (x <= 2.0) else np.nan
        )
        self.__fillna__(self.user_logs_df,'num_75',0.0)
        
        # N985 - Handle Outliers then Fill w/Median
        self.user_logs_df['num_985'] = self.user_logs_df['num_985'].apply(
            lambda x: x if (x <= 2.0) else np.nan
        )
        self.__fillna__(self.user_logs_df,'num_985',0.0)
        
        # N100 - Handle Outliers then Fill w/Median
        self.user_logs_df['num_100'] = self.user_logs_df['num_100'].apply(
            lambda x: x if (x <= 82.0) else np.nan
        )
        self.__fillna__(self.user_logs_df,'num_100',17.0)

        
        # NUnq - Handle Outliers then Fill w/Median
        self.user_logs_df['num_unq'] = self.user_logs_df['num_unq'].apply(
            lambda x: x if (x <= 83.0) else np.nan
        )
        self.__fillna__(self.user_logs_df,'num_unq',18.0)
        
        # Total Secs - Handle Outliers then Fill w/Median
        self.user_logs_df['total_secs'] = self.user_logs_df['total_secs'].apply(
            lambda x: x if (x <= 21562.0) else np.nan
        )
        self.__fillna__(self.user_logs_df,'total_secs',4548.0)

        

    
    
   
