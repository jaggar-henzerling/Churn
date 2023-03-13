import pandas as pd
import dask.dataframe as dd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from src import processing as pr
from src import transactions_features as tf
from src import user_logs_features as uf

# Generic dataset handler
class Dataset(pr.Preprocessor, tf.TransactionsFeatures, uf.UserLogsFeatures):
    # Init, specifies name and holds locations
    def __init__(self, which):
        
        # Set the data locations
        data_dir = os.getcwd() + "/data/"
        self.user_logs = data_dir + "user_logs_v2.csv"
        self.transactions = data_dir + "transactions_v2.csv"
        self.members = data_dir + "members_v3.csv"
        
        # Also include save name (if need to save the data)
        if which == "train":
            self.data_which = data_dir + "train_v2.csv"
            self.save_name = data_dir + "data_train.csv"
        elif which == "test":
            self.data_which = data_dir + "sample_submission_v2.csv"
            self.save_name = data_dir + "data_test.csv"
        else:
            print("BAD DATA NAME")
            
    
    # Initialize Dataset (Public method, stores in object)
    def initialize(self):

        self.__load_data__()
        
        # Pre Merge so that MSNOs are corroborated
        self.__pre_merge__()
        
        # Process Raw Data Before Feature Engineering
        self.__preprocess__()
        
        # Feature Engineering - User_Logs
        self.user_logs_df = super().user_logs_features()
        
        # Feature Engineering - Transactions
        self.transactions_df = super().transactions_features()
        
        # Merge the data, keep the columns for utility, then clear the old data
        self.__merge_data__()
        self.__keep_columns__()
        self.__clear_old_data__()
        
    # Save the data
    def save_data(self):
        pd.to_csv(self.save_name)
        
    # Load the data from memory - used to not need to initialize on every run
    def load_data(self):
        self.data = pd.read_csv(self.save_name)
        
    # Run preprocessing
    def __preprocess__(self):
        super().preprocess()
            
        
    # Load the data in (PANDAS) - Done so that loading is frontloaded
    def __load_data__(self):
        self.user_logs_df = pd.read_csv(self.user_logs)
        self.transactions_df = pd.read_csv(self.transactions)
        self.members_df = pd.read_csv(self.members)
        self.data_df = pd.read_csv(self.data_which)        
    
    # Pre-merge, this just guarantees that we catch when msnos aren't in certain datasets
    # Drop is_churn since otherwise we add too many columns
    def __pre_merge__(self):
        self.user_logs_df = self.data_df.merge(self.user_logs_df, how='left', on='msno').drop(columns=['is_churn'])
        self.members_df = self.data_df.merge(self.members_df, how='left', on='msno').drop(columns=['is_churn'])
        self.transactions_df = self.data_df.merge(self.transactions_df, how='left', on='msno').drop(columns=['is_churn'])
    
    # Merge data and store
    def __merge_data__(self):
        self.data = self.data_df.merge(self.user_logs_df, how='left', on='msno')
        self.data = self.data.merge(self.members_df, how='left', on='msno')
        self.data = self.data.merge(self.transactions_df, how='left', on='msno')        
        
    # Clear old data for memory efficiency
    def __clear_old_data__(self):
        del self.user_logs_df
        del self.transactions_df
        del self.members_df
        del self.data_df
        
    # Store column groupings
    def __keep_columns__(self):
        self.user_logs_columns = self.user_logs_df.columns.tolist()
        self.members_columns = self.members_df.columns.tolist()
        self.transactions_columns = self.transactions_df.columns.tolist()
        self.data_columns = self.data_df.columns.tolist()
        
        
