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
import json

def calc_level_ofi(df, level=0):
    """
    Compute multi-level OFI for a certain level
    """
    df = df.sort_values(by=['instrument_id', 'ts_event']).reset_index(drop=True)
    group = df.groupby('instrument_id')
    
    bid_px = df[f'bid_px_{level:02d}']
    bid_sz = df[f'bid_sz_{level:02d}']
    ask_px = df[f'ask_px_{level:02d}']
    ask_sz = df[f'ask_sz_{level:02d}']
    
    shifted_bid_px = group[f'bid_px_{level:02d}'].shift(1)
    shifted_bid_sz = group[f'bid_sz_{level:02d}'].shift(1)
    shifted_ask_px = group[f'ask_px_{level:02d}'].shift(1)
    shifted_ask_sz = group[f'ask_sz_{level:02d}'].shift(1)
    

    of_bid = np.where(bid_px > shifted_bid_px, bid_sz,
                      np.where(bid_px < shifted_bid_px, -bid_sz,
                               bid_sz - shifted_bid_sz))
    
    of_ask = np.where(ask_px > shifted_ask_px, -ask_sz,
                      np.where(ask_px < shifted_ask_px, ask_sz,
                               ask_sz - shifted_ask_sz))
    
    df[f'ofi_{level}'] = of_bid - of_ask
    return df

def calc_ofi_for_levels(df, levels=5):
    """
    Compute multi-level OFI for levels upto levels
    """
    df_out = df.copy()
    for l in range(levels):
        df_out = calc_level_ofi(df_out, level=l)
    return df_out

def calculate_lagged_ofi(df):
    """
    Compute lagged ofi, shift by a single event
    """
    df['ofi_lagged'] = df.groupby('instrument_id')['ofi_pca'].shift(1)
    return df


def assign_minute_bucket(df):
    """
    Assign each event to a 1-minute bucket,
    and compute the mid-price at each event.
    """
    df['datetime'] = pd.to_datetime(df['ts_event'], unit='ns')
    df['minute'] = df['datetime'].dt.floor('T')
    df['mid_price'] = (df['bid_px_00'] + df['ask_px_00']) / 2
    return df

def aggregate_ofi_minute_with_scaling(df, levels=5):
    """
    Aggregate data over 1-minute intervals, scale OFI by average depth, 
    and compute integrated OFI using PCA.
    """
        
    #define aggregation: sum OFI and sizes, count events, and get last mid_price
    agg_dict = {'mid_price': 'last', 'ts_event': 'count'}
    for l in range(levels):
        agg_dict[f'ofi_{l}'] = 'sum'
        agg_dict[f'bid_sz_{l:02d}'] = 'sum'
        agg_dict[f'ask_sz_{l:02d}'] = 'sum'
        
    grouped = df.groupby(['instrument_id', 'minute']).agg(agg_dict).reset_index()
    
    #compute average size and scale OFI for each level
    for l in range(levels):
        bid_sum = grouped[f'bid_sz_{l:02d}']
        ask_sum = grouped[f'ask_sz_{l:02d}']
        N = grouped['ts_event']  # count of events in the minute
        grouped[f'avg_size_{l}'] = (bid_sum + ask_sum) / (2 * N.replace(0, np.nan))
    
    #compute QM as the average over levels
    avg_sizes = [grouped[f'avg_size_{l}'] for l in range(levels)]
    grouped['QM'] = np.mean(np.column_stack(avg_sizes), axis=1)
    
    #scale OFI for each level using QM
    for l in range(levels):
        grouped[f'scaled_ofi_{l}'] = grouped[f'ofi_{l}'] / grouped['QM']
    
    #compute integrated OFI using PCA on scaled OFI columns
    ofi_cols = [f'scaled_ofi_{l}' for l in range(levels)]
    df_for_pca = grouped.dropna(subset=ofi_cols)
    pca = PCA(n_components=1)
    pca.fit(df_for_pca[ofi_cols])
    w1 = pca.components_[0]
    l1_norm = np.sum(np.abs(w1))
    w1_normalized = w1 / l1_norm
    w11 = w1_normalized[0] if w1_normalized[0] != 0 else 1.0
    df_for_pca['ofi_pca'] = df_for_pca[ofi_cols].dot(w1_normalized) / w11
    
    return df_for_pca


def compute_future_price_changes(aggregated_df, horizons=[1,5]):
    """
    Compute price change metrisc
    """
    
    aggregated_df = aggregated_df.sort_values(['instrument_id', 'minute']).copy()
    
    aggregated_df['price_change'] = aggregated_df.groupby('instrument_id')['mid_price'].diff()
    
    aggregated_df['log_return'] = np.log(
        aggregated_df['mid_price'] / aggregated_df.groupby('instrument_id')['mid_price'].shift(1)
    )
    
    for horizon in horizons:
        price_col_name = f'future_price_change_{horizon}m'
        log_col_name = f'future_log_return_{horizon}m'
        
        aggregated_df[price_col_name] = (
            aggregated_df.groupby('instrument_id')['mid_price']
            .shift(-horizon) - aggregated_df['mid_price']
        )
        
        aggregated_df[log_col_name] = np.log(
            aggregated_df.groupby('instrument_id')['mid_price'].shift(-horizon) / aggregated_df['mid_price']
        )
    
    return aggregated_df


