import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import pearsonr, spearmanr
from alpha_provider import AlphaFactory
import streamlit as st

class QlibWorkflow:
    def __init__(self, data_loader_func):
        """
        data_loader_func: Function that returns dict {ticker: DataFrame...} or raw DataFrame
        """
        self.data_loader = data_loader_func
        self.models = {}
        
    def prepare_data(self, tickers, start_date, end_date):
        """
        1. Load Raw Data
        2. Generate Alphas (Features)
        3. Generate Labels
        4. Return Aligned DataFrame (MultiIndex: Date, Ticker)
        """
        # 1. Load Data (Reuse app's logic via injected function or simple loop)
        # Assuming data_loader returns dictionary {ticker: df}
        raw_data_dict = self.data_loader(tickers, start_date, end_date)
        
        if not raw_data_dict:
            return None, "No data loaded."
            
        # Merge into Single DataFrame for AlphaFactory (MultiIndex Efficient)
        # DF Columns: [Open, High, Low, Close, Volume]
        # Index: [Date, Ticker] (or Ticker column)
        
        frames = []
        for t, df in raw_data_dict.items():
            df = df.copy()
            df['Ticker'] = t
            # Ensure columns are standardized
            # Handle potential case mismatches
            cols_map = {c: c.capitalize() for c in df.columns} 
            # Force standard names if possible, but yfinance is usually Title Case
            # AlphaFactory expects Title Case: Open, High, Low, Close, Volume
            frames.append(df)
            
        if not frames:
            return None, "Empty dataframes."
            
        full_df = pd.concat(frames)
        
        # Normalize Index Name
        full_df.index.name = 'Date'
        
        # Reset Index to column
        full_df = full_df.reset_index()
        
        # Normalize Date Column (Remove Time zone/Time)
        if 'Date' in full_df.columns:
            full_df['Date'] = pd.to_datetime(full_df['Date']).dt.normalize()
            
        # Deduplicate (Strict)
        # Sort to keep latest
        if 'Date' in full_df.columns and 'Ticker' in full_df.columns:
            full_df = full_df.sort_values(['Date', 'Ticker'])
            full_df = full_df.drop_duplicates(subset=['Date', 'Ticker'], keep='last')
        else:
            return None, "Columns 'Date' and 'Ticker' required after processing."
        
        # Set MultiIndex [Date, Ticker]
        full_df = full_df.set_index(['Date', 'Ticker']).sort_index()
        
        # Verify Uniqueness
        if not full_df.index.is_unique:
             # Force deduplication on index
             full_df = full_df[~full_df.index.duplicated(keep='last')]
        
        # 2. Generate Alphas
        
        # 2. Generate Alphas
        try:
            alphas = AlphaFactory.get_alpha158_lite(full_df)
        except Exception as e:
            return None, f"Alpha Generation Failed: {e}"
            
        # 3. Generate Labels (Next Day Return)
        # Horizon 1 day usually
        labels = AlphaFactory.get_labels(full_df, horizon=1)
        
        # 4. Merge
        # Alphas and Labels share the same index (Date, Ticker)
        dataset = pd.concat([alphas, labels], axis=1)
        dataset = dataset.dropna() # Drop rows with missing features or labels (last day has no label)
        
        return dataset, None

    def train_model(self, train_df, val_df, feature_cols, label_col='Ref($close, -1)', **kwargs):
        """
        Train LightGBM Ranking/Regression Model
        """
        X_train = train_df[feature_cols]
        y_train = train_df[label_col]
        
        X_val = val_df[feature_cols]
        y_val = val_df[label_col]
        
        # Default Params
        params = {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'random_state': 42,
            'n_jobs': -1
        }
        # Update with kwargs
        params.update(kwargs)
        
        # LGBM Regressor (Predicting Return)
        model = lgb.LGBMRegressor(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='mse',
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False)
            ]
        )
        
        return model

    def analyze_performance(self, model, test_df, feature_cols, label_col='Ref($close, -1)'):
        """
        Calculate IC, Rank IC
        """
        X_test = test_df[feature_cols]
        y_test = test_df[label_col]
        
        preds = model.predict(X_test)
        
        # Add predictions to dataframe to group by Date
        res_df = test_df[[label_col]].copy()
        res_df['Pred'] = preds
        
        # Calculate IC per day
        # IC = Correlation(Pred, Label)
        daily_ic = res_df.groupby(level='Date').apply(
            lambda x: pearsonr(x['Pred'], x[label_col])[0] if len(x) > 2 else np.nan
        )
        
        # Rank IC = Spearman Correlation(Pred, Label)
        daily_rank_ic = res_df.groupby(level='Date').apply(
            lambda x: spearmanr(x['Pred'], x[label_col])[0] if len(x) > 2 else np.nan
        )
        
        metrics = {
            "IC_Mean": daily_ic.mean(),
            "IC_Std": daily_ic.std(),
            "ICIR": daily_ic.mean() / (daily_ic.std() + 1e-9),
            "Rank_IC_Mean": daily_rank_ic.mean(),
            "Rank_IC_Std": daily_rank_ic.std(),
            "Rank_ICIR": daily_rank_ic.mean() / (daily_rank_ic.std() + 1e-9)
        }
        
        return metrics, daily_ic, daily_rank_ic, res_df
