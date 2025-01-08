
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

def predict_price_change_with_lasso(aggregated_df, target_instrument, predictor_col, target_col, cv=5, random_state=0):
    """
    Predicts price change of a target instrument using LASSO regression.
    """
    ofi_pivot = aggregated_df.pivot(index='minute', columns='instrument_id', values=predictor_col)

    target_data = aggregated_df.loc[aggregated_df['instrument_id'] == target_instrument, ['minute', target_col]]
    data_merged = ofi_pivot.join(target_data.set_index('minute'), how='inner').dropna()

    X, y = data_merged.drop(columns=target_col), data_merged[target_col]

    lasso = LassoCV(cv=cv, random_state=random_state).fit(X, y)

    coef = pd.Series(lasso.coef_, index=X.columns)[lambda x: x != 0]

    return coef, lasso.intercept_, lasso.score(X, y)

