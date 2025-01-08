import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import databento as db
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import seaborn as sns

def raw_data_processing(df):
    """
    Handling raw data and removing unused columns and preparing data for analysis
    """
    df = df.drop(columns=['bid_sz_05', 'ask_sz_05', 'bid_ct_05', 'ask_ct_05',
        'bid_sz_06', 'ask_sz_06', 'bid_ct_06', 'ask_ct_06', 'bid_sz_07',
        'ask_sz_07', 'bid_ct_07', 'ask_ct_07', 'bid_sz_08', 'ask_sz_08',
        'bid_ct_08', 'ask_ct_08', 'bid_sz_09', 'ask_sz_09', 'bid_ct_09',
        'ask_ct_09', 'bid_px_05', 'ask_px_05', 'bid_px_06', 'ask_px_06', 'bid_px_07', 'ask_px_07', 'bid_px_08', 'ask_px_08', 'bid_px_09', 'ask_px_09'])

    #keeping only ask and bid events
    df = df[df['side'].isin(['B', 'A'])].reset_index(drop=True)
    price_columns = [col for col in df.columns if 'price' in col or 'bid_px' in col or 'ask_px' in col]
    df[price_columns] = df[price_columns] * 1e-9

    return df

def clean(df):
    """
    clean data by removing nan and inf values
    """
    wanted = ['log_return', 'price_change', "ofi_pca", "ofi_lagged", "future_price_change_5m", "future_price_change_1m"]
    existing_columns = [col for col in wanted if col in df.columns]
    
    columns_to_drop = [col for col in df.columns if 'ask' in col or 'bid' in col]
    cleaned = df.drop(columns=columns_to_drop)
    
    cleaned[existing_columns] = cleaned[existing_columns].replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(subset=existing_columns)
    
    return cleaned
  