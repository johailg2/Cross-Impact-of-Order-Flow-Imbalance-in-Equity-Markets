
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

def regression(data, y, x):
    df = data[[y] + x].copy()
    X = sm.add_constant(df[x])
    Y = df[y]
    model = sm.OLS(Y, X).fit()
    return model

def cross_impact_analysis(data):
    """
    Performs cross impact correlation matrix calcs.
    """

    ids = data["instrument_id"].unique()
    
    impact_matrix = pd.DataFrame(index=ids, columns=ids, dtype=float)
    
    ofi_data = {sym: data[data["instrument_id"] == sym]['ofi_pca'].values for sym in ids}
    price_data = {sym: data[data["instrument_id"] == sym]['price_change'].values for sym in ids}
    
    for id in ids:
        ofi1 = ofi_data[id]
        for id2 in ids:
            ofi2 = price_data[id2]
            min_len = min(len(ofi1), len(ofi2))
            corr = np.corrcoef(ofi1[:min_len], ofi2[:min_len])[0, 1]
            impact_matrix.at[id, id2] = corr if np.isfinite(corr) else np.nan

    return impact_matrix

def regression_cross_impact(returns_series, ofis, mapping_dict):
    """
    Perform OLS regression on one stocks returns using other stock's ofis
    """
    ofis.rename(columns=mapping_dict, inplace=True)
    
    df = ofis.copy()
    df['log_return'] = returns_series
    
    df = df.dropna(subset=['log_return'])
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    X = sm.add_constant(df[ofis.columns])
    y = df['log_return']
    
    model = sm.OLS(y, X).fit()
    return model


