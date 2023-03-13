import pandas as pd
import dask.dataframe as dd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Class which handles generating new features and appending to DF - specifically for the transactions set
class TransactionsFeatures:
        
    # Main feature method to be run, generates and appends statistical features for models
    def transactions_features(self):
        
        # Aggregate features based on the MSNO
        tagg = self.transactions_df.groupby('msno').agg(
            recent_payment_method = ('payment_method_id', 'last'),
            payment_plan_mean = ('payment_plan_days','mean'),
            plan_list_price_mean = ('plan_list_price','mean'),
            actual_amount_paid_mean = ('actual_amount_paid','mean'),
            recent_auto_renew = ('is_auto_renew','last'),
            n_transactions = ('transaction_date','count'),
            recent_transaction = ('transaction_date','max'),
            recent_expiry = ('membership_expire_date','max'),
            recent_cancel = ('is_cancel','last')
        )
        
        # Non robust discount
        tagg['discount_mean'] = (tagg['plan_list_price_mean'] - tagg['actual_amount_paid_mean']).clip(lower=0)
        
        # Loyalty (days) (UNIMPLEMENTED)
        #tagg['loyalty'] = 
        
        # Set the transactions to the new aggregate
        return tagg
    
    
    
    
