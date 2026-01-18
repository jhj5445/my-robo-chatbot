import pandas as pd
import numpy as np

class AlphaFactory:
    """
    Qlib-Lite Alpha Factory.
    Replicates key operators and factors from Qlib's expression engine using pure Pandas.
    Expects input DataFrame with MultiIndex (Date, Ticker) and columns: [Open, High, Low, Close, Volume].
    """
    
    @staticmethod
    def _validate(df):
        # Ensure MultiIndex
        if not isinstance(df.index, pd.MultiIndex):
            # Try to set index if columns exist
            if 'Date' in df.columns and 'Ticker' in df.columns:
                return df.set_index(['Date', 'Ticker']).sort_index()
            else:
                raise ValueError("Input DataFrame must be MultiIndex(Date, Ticker) or have Date/Ticker columns.")
        return df

    # --- Operators (Qlib Mimic) ---
    @staticmethod
    def Ref(series, d):
        return series.groupby(level='Ticker').shift(d)

    @staticmethod
    def Mean(series, d):
        return series.groupby(level='Ticker').rolling(d).mean().reset_index(0, drop=True)

    @staticmethod
    def Std(series, d):
        return series.groupby(level='Ticker').rolling(d).std().reset_index(0, drop=True)

    @staticmethod
    def Max(series, d):
        return series.groupby(level='Ticker').rolling(d).max().reset_index(0, drop=True)

    @staticmethod
    def Min(series, d):
        return series.groupby(level='Ticker').rolling(d).min().reset_index(0, drop=True)
    
    @staticmethod
    def Delta(series, d):
        return series - AlphaFactory.Ref(series, d)

    @staticmethod
    def Sma(series, d):
        # Simple Moving Average (Same as Mean)
        return AlphaFactory.Mean(series, d)
        
    @staticmethod
    def Wma(series, d):
        # Weighted Moving Average (Linear weights)
        def linear_wma(x):
            weights = np.arange(1, len(x) + 1)
            return np.dot(x, weights) / weights.sum()
        return series.groupby(level='Ticker').rolling(d).apply(linear_wma, raw=True).reset_index(0, drop=True)

    @staticmethod
    def Corr(series_a, series_b, d):
        # Rolling Correlation
        return series_a.groupby(level='Ticker').rolling(d).corr(series_b).reset_index(0, drop=True)

    # --- Alpha Factors (Subset of Alpha158) ---
    @classmethod
    def get_alpha158_lite(cls, df):
        """
        Generates a lite version of Alpha158 (Top 20 widely used factors).
        """
        df = cls._validate(df)
        O = df['Open']
        H = df['High']
        L = df['Low']
        C = df['Close']
        V = df['Volume']
        
        # VWAP (Approximate: (H+L+C)/3 * V / V_sum?) 
        # Actually daily vwap is ~ (O+H+L+C)/4 or Amount/Volume. 
        # Using (O+H+L+C)/4 as proxy for daily average price if Amount missing.
        AvgPrice = (O + H + L + C) / 4
        
        alphas = pd.DataFrame(index=df.index)
        
        # 1. KMID (K-Mid): (Close - Open) / Open
        alphas['KMID'] = (C - O) / O
        
        # 2. KLEN (K-Len): (High - Low) / Open
        alphas['KLEN'] = (H - L) / O
        
        # 3. KMID2: (Close - Open) / (High - Low + 1e-8)
        alphas['KMID2'] = (C - O) / (H - L + 1e-8)
        
        # 4. KUP: (High - Max(Open, Close)) / Open
        # Use numpy maximum for safety and speed
        max_oc = np.maximum(O, C)
        alphas['KUP'] = (H - max_oc) / O
        
        # 5. KLOW: (Min(Open, Close) - Low) / Open
        min_oc = np.minimum(O, C)
        alphas['KLOW'] = (min_oc - L) / O
        
        # 6. KSFT (K-Shift): (2*Close - High - Low) / Open
        alphas['KSFT'] = (2 * C - H - L) / O
        
        # 7. KOPT (K-Opt): (Close - Previous Close) / Open
        prev_close = cls.Ref(C, 1)
        alphas['KOPT'] = (C - prev_close) / O
        
        # 8. ROC_5, ROC_10, ROC_20, ROC_60
        for d in [5, 10, 20, 60]:
            alphas[f'ROC{d}'] = cls.Ref(C, d) / C - 1 # Note: Qlib ROC is usually Ref(C, d)/C or C/Ref(C,d) - 1. 
            # Using C / Ref - 1 (Standard Return).
            alphas[f'ROC{d}'] = C / cls.Ref(C, d) - 1
            
        # 9. MA_Dist (Price / MA - 1)
        for d in [5, 10, 20, 60]:
            ma = cls.Mean(C, d)
            alphas[f'MA{d}'] = C / ma - 1
            
        # 10. Volatility (Std / Close)
        for d in [20, 60]:
            std = cls.Std(C, d)
            alphas[f'STD{d}'] = std / C
            
        # 11. Beta / Correlation Proxy (Vol / Price Correlation)
        # Corr(Close, Volume, 20)
        alphas['CORD20'] = cls.Corr(C, V, 20)
        alphas['CORD60'] = cls.Corr(C, V, 60)
        
        # 12. VMA (Volume MA Ratio)
        for d in [5, 20]:
            vma = cls.Mean(V, d)
            alphas[f'VMA{d}'] = V / vma - 1
            
        # 13. RSI (Relative Strength Index)
        # Using simple method for robustness
        delta = C.diff()
        # Need vectorized groupby shift for diff? C is already aligned but diff crosses tickers?
        # No, diff() on Series respects index order. But C is MultiIndex.
        # Must use GroupBy Diff
        delta = C.groupby(level='Ticker').diff()
        
        for d in [14]:
            up, down = delta.copy(), delta.copy()
            up[up < 0] = 0
            down[down > 0] = 0
            
            roll_up = up.groupby(level='Ticker').rolling(d).mean().reset_index(0, drop=True)
            roll_down = down.abs().groupby(level='Ticker').rolling(d).mean().reset_index(0, drop=True)
            
            rs = roll_up / roll_down
            alphas[f'RSI{d}'] = 100.0 - (100.0 / (1.0 + rs))
            
        # Clean Infinite/NaN
        alphas.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Fill NaN (Simple Forward Fill then 0)
        alphas = alphas.groupby(level='Ticker').ffill().fillna(0)
        
        return alphas

    # --- Label Generation ---
    @staticmethod
    def get_labels(df, horizon=1):
        """
        Generates Ref($close, -d) / $close - 1
        """
        df = AlphaFactory._validate(df)
        C = df['Close']
        
        # Next Return: Close_t+h / Close_t - 1
        # Shift(-h) brings future value to current row
        future_close = C.groupby(level='Ticker').shift(-horizon)
        label = future_close / C - 1
        
        return label.rename(f"Ref($close, -{horizon})")
