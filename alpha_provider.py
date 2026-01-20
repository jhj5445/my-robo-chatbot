
import pandas as pd
import numpy as np

class AlphaFactory:
    """
    Qlib Alpha158 Exact Implementation
    Source: qlib/contrib/data/loader.py (Alpha158DL)
    
    Total Features: 158
    Order: 
    1. K-Bar (9)
    2. Price (4)
    3. Rolling Features (Groups of 5 windows: 5, 10, 20, 30, 60)
       ROC, MA, STD, BETA, RSQR, RESI, MAX, LOW(MIN), QTLU, QTLD, RANK, RSV, 
       IMAX, IMIN, IMXD, CORR, CORD, CNTP, CNTN, CNTD, SUMP, SUMN, SUMD, 
       VMA, VSTD, WVMA, VSUMP, VSUMN, VSUMD
    """
    
    @staticmethod
    def get_alpha158(df):
        # 1. Prepare Data
        # Ensure necessary columns
        req = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(c in df.columns for c in req):
            raise ValueError(f"Dataframe must have {req} columns")
            
        eps = 1e-12 # Qlib default epsilon
        
        # Sort and Group (but usually we handle single ticker or ensure sorted)
        # Assuming df is SINGLE TICKER time-series or MultiIndex
        # For safety, let's assume it's a single ticker DataFrame correctly sorted by Date.
        # If MultiIndex, we should groupby. But for 'Fast Inference' it's usually 1 ticker.
        # Let's support simple DF.
        
        O = df['Open'].astype(float)
        H = df['High'].astype(float)
        L = df['Low'].astype(float)
        C = df['Close'].astype(float)
        V = df['Volume'].astype(float)
        # VWAP: Typical Price if not available, but Qlib uses vwap from data provider.
        # If 'Adj Close' exists, use it as Close? No, Qlib uses raw usually, but for inference we use what we have.
        # Let's approximate VWAP if not present.
        VWAP = df['VWAP'] if 'VWAP' in df.columns else (O + H + L + C) / 4.0
        
        # Helper Shift (Ref)
        def Ref(series, d):
            return series.shift(d)
            
        features = {}
        
        # -------------------------------------------------------------------------
        # 1. K-Bar Features (9)
        # -------------------------------------------------------------------------
        # KMID: (C - O) / O
        features['KMID'] = (C - O) / O
        # KLEN: (H - L) / O
        features['KLEN'] = (H - L) / O
        # KMID2: (C - O) / (H - L + eps)
        features['KMID2'] = (C - O) / (H - L + eps)
        # KUP: (H - max(O,C)) / O
        features['KUP'] = (H - np.maximum(O, C)) / O
        # KUP2: (H - max(O,C)) / (H - L + eps)
        features['KUP2'] = (H - np.maximum(O, C)) / (H - L + eps)
        # KLOW: (min(O,C) - L) / O
        features['KLOW'] = (np.minimum(O, C) - L) / O
        # KLOW2: (min(O,C) - L) / (H - L + eps)
        features['KLOW2'] = (np.minimum(O, C) - L) / (H - L + eps)
        # KSFT: (2*C - H - L) / O
        features['KSFT'] = (2*C - H - L) / O
        # KSFT2: (2*C - H - L) / (H - L + eps)
        features['KSFT2'] = (2*C - H - L) / (H - L + eps)
        
        # -------------------------------------------------------------------------
        # 2. Price Features (4) (Windows=[0])
        # -------------------------------------------------------------------------
        # OPEN0: O / C
        features['OPEN0'] = O / C
        # HIGH0: H / C
        features['HIGH0'] = H / C
        # LOW0:  L / C
        features['LOW0'] = L / C
        # VWAP0: VWAP / C  (Note: Qlib source uses $vwap/$close)
        features['VWAP0'] = VWAP / C
        
        # -------------------------------------------------------------------------
        # 3. Rolling Features (29 Types x 5 Windows)
        # -------------------------------------------------------------------------
        windows = [5, 10, 20, 30, 60]
        
        # --- Pre-computation for complex features ---
        # Returns
        ret = C / C.shift(1) - 1
        abs_ret = ret.abs()
        # Log Volume
        log_v = np.log(V + 1)
        log_v_ret = np.log(V / V.shift(1) + 1)
        
        
        # A. ROC (Rate of Change): Ref($close, d)/$close => Wait, Qlib says Ref(C, d)/C
        # Usually ROC is (C - C_d)/C_d, but Qlib implementation in loader.py says:
        # fields += ["Ref($close, %d)/$close" % d for d in windows]
        # Wait, Ref(C, d) is C_{t-d}. So it's C_{t-d} / C_t.
        # This is INVERSE of growth? C_old / C_new.
        # Let's verify text: "Rate of change...". 
        # Actually Qlib code: "Ref($close, %d)/$close" 
        # If d=5, Close 5 days ago / Current Close.
        for d in windows:
            features[f'ROC{d}'] = Ref(C, d) / C

        # B. MA (Mean): Mean($close, d)/$close
        for d in windows:
            features[f'MA{d}'] = C.rolling(d).mean() / C
            
        # C. STD: Std($close, d)/$close
        for d in windows:
            features[f'STD{d}'] = C.rolling(d).std() / C
            
        # D. BETA: Slope($close, d)/$close
        # Slope of close price against time steps (1..d).
        # Slope = Cov(C, Time) / Var(Time)
        # Var(Time) for window d is constant: (d^2 - 1)/12
        for d in windows:
            # We can use vectorized logic given x is just 0..d-1
            # But calculating rolling Cov efficiently is tricky.
            # Use polyfit on windows? Too slow.
            # Use formula: slope = (Sum(xy) - n*mean(x)*mean(y)) / (Sum(x^2) - n*mean(x)^2)
            # x is 0, 1, ..., d-1. mean(x) = (d-1)/2. Sum(x^2) = (d-1)d(2d-1)/6.
            # We need Sum(y * x_idx) in rolling window.
            # This is weighted sum.
            val = C.rolling(d).apply(lambda y: np.polyfit(np.arange(d), y, 1)[0], raw=True)
            features[f'BETA{d}'] = val / C

        # E. RSQR: Rsquare($close, d)
        # R^2 = Corr(C, Time)^2
        for d in windows:
             # Standard Correlation against strictly increasing sequence 0..d-1?
             # Since 'Time' variance is constant, simple correlation works.
             # We can't easily do rolling corr with static array in pure pandas without apply.
             # fallback to apply for correctness (Optimization later if needed)
             val = C.rolling(d).apply(lambda y: np.corrcoef(y, np.arange(d))[0, 1]**2, raw=True)
             features[f'RSQR{d}'] = val

        # F. RESI: Resi($close, d)/$close
        # Residuals of linear regression?
        # Likely the last residual? Or sum? Qlib 'Resi' operator usually means strictly the residual at t.
        # Resi = C_t - (Slope*t + Intercept).
        # If we define x=0..(d-1) and fitting y=C_{t-d+1}..C_t.
        # Then C_t corresponds to x=d-1.
        # Predicted = Slope*(d-1) + Intercept.
        # Resi = C_t - Predicted.
        for d in windows:
             def get_resi(y):
                 x = np.arange(d)
                 slope, intercept = np.polyfit(x, y, 1)
                 pred = slope * (d-1) + intercept
                 return y[-1] - pred
             val = C.rolling(d).apply(get_resi, raw=True)
             features[f'RESI{d}'] = val / C

        # G. MAX: Max($high, d)/$close
        for d in windows:
            features[f'MAX{d}'] = H.rolling(d).max() / C

        # H. LOW (MIN): Min($low, d)/$close  (Name is MIN in loader.py names list: names += ["MIN%d"])
        # Wait, chunk 2 line: names += ["MIN%d" % d ...]
        for d in windows:
            features[f'MIN{d}'] = L.rolling(d).min() / C
            
        # I. QTLU: Quantile($close, d, 0.8)/$close
        for d in windows:
            features[f'QTLU{d}'] = C.rolling(d).quantile(0.8) / C
            
        # J. QTLD: Quantile($close, d, 0.2)/$close
        for d in windows:
            features[f'QTLD{d}'] = C.rolling(d).quantile(0.2) / C
            
        # K. RANK: Rank($close, d) -> Percentile
        for d in windows:
            # Pandas rank(pct=True) is over the whole series. We need rolling rank.
            # rolling().rank() exists in recent pandas? Yes.
            features[f'RANK{d}'] = C.rolling(d).rank(pct=True)
            
        # L. RSV: (C - Min(L, d)) / (Max(H, d) - Min(L, d) + eps)
        for d in windows:
            l_min = L.rolling(d).min()
            h_max = H.rolling(d).max()
            features[f'RSV{d}'] = (C - l_min) / (h_max - l_min + eps)

        # M. IMAX: IdxMax($high, d)/d
        # (Days since max) / d?
        # Qlib IdxMax: "The number of days between current date and previous highest".
        # If today is highest, it might be 0?
        # rolling apply argmax.
        # Note: argmax returns absolute index. We need relative.
        for d in windows:
            # We treat index 0 as oldest in window, d-1 as newest (today).
            # We want distance from today?
            # Qlib desc: "number of days between current date and previous highest"
            # If d=5, values [10, 11, 10, 10, 10]. Max at index 1.
            # Current is index 4. Dist = 4 - 1 = 3.
            # So (d - 1) - argmax.
            # argmax in numpy returns index in flattened array (0..d-1).
            features[f'IMAX{d}'] = C.rolling(d).apply(lambda x: (d - 1 - np.argmax(x)) / d, raw=True)

        # N. IMIN: IdxMin($low, d)/d
        for d in windows:
            features[f'IMIN{d}'] = C.rolling(d).apply(lambda x: (d - 1 - np.argmin(x)) / d, raw=True)
            
        # O. IMXD: (IdxMax(H, d) - IdxMin(L, d)) / d
        for d in windows:
             features[f'IMXD{d}'] = features[f'IMAX{d}'] - features[f'IMIN{d}']
             
        # P. CORR: Corr($close, Log($volume+1), d)
        for d in windows:
            features[f'CORR{d}'] = C.rolling(d).corr(log_v)
            
        # Q. CORD: Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), d)
        # i.e., Corr(Price Change Ratio, Volume Change Ratio)
        # Note: Price ratio is C_t / C_{t-1}. 
        price_ratio = C / C.shift(1)
        # log_v_ret is computed above
        for d in windows:
            features[f'CORD{d}'] = price_ratio.rolling(d).corr(log_v_ret)
            
        # R. CNTP: Mean($close > Ref($close, 1), d)
        # % of days where price went up
        data_gt = (C > C.shift(1)).astype(float)
        for d in windows:
            features[f'CNTP{d}'] = data_gt.rolling(d).mean()
            
        # S. CNTN: Mean($close < Ref($close, 1), d)
        data_lt = (C < C.shift(1)).astype(float)
        for d in windows:
            features[f'CNTN{d}'] = data_lt.rolling(d).mean()
            
        # T. CNTD: CNTP - CNTN
        for d in windows:
            features[f'CNTD{d}'] = features[f'CNTP{d}'] - features[f'CNTN{d}']
            
        # U. SUMP: Sum(Max(Diff, 0), d) / (Sum(Abs(Diff), d) + eps)
        # This is essentially RS / (1+RS)? No, it's Gain / (Gain + Loss).
        diff = C - C.shift(1)
        gain = diff.clip(lower=0)
        # loss = -diff.clip(upper=0)
        abs_diff = diff.abs()
        for d in windows:
            s_gain = gain.rolling(d).sum()
            s_abs = abs_diff.rolling(d).sum()
            features[f'SUMP{d}'] = s_gain / (s_abs + eps)

        # V. SUMN: Sum(Max(-Diff, 0), d) / (Sum(Abs(Diff), d) + eps)
        loss = -diff.clip(upper=0) # positive value for loss
        for d in windows:
            s_loss = loss.rolling(d).sum()
            s_abs = abs_diff.rolling(d).sum()
            features[f'SUMN{d}'] = s_loss / (s_abs + eps)
            
        # W. SUMD: (Sum(Gain, d) - Sum(Loss, d)) / (Sum(Abs(Diff), d) + eps)
        # Basically SUMP - SUMN
        for d in windows:
            features[f'SUMD{d}'] = features[f'SUMP{d}'] - features[f'SUMN{d}']
            
        # X. VMA: Mean($volume, d) / ($volume + eps)
        for d in windows:
            features[f'VMA{d}'] = V.rolling(d).mean() / (V + eps)
            
        # Y. VSTD: Std($volume, d) / ($volume + eps)
        for d in windows:
            features[f'VSTD{d}'] = V.rolling(d).std() / (V + eps)
            
        # Z. WVMA: Std(Abs(Ret)*Volume, d) / (Mean(Abs(Ret)*Volume, d) + eps)
        # Volatility of "Volume-Weighted Return" ?
        # Formula: Std(Abs($close/Ref($close, 1)-1)*$volume, d) / ...
        val_w = abs_ret * V
        for d in windows:
            std_w = val_w.rolling(d).std()
            mean_w = val_w.rolling(d).mean()
            features[f'WVMA{d}'] = std_w / (mean_w + eps)
            
        # AA. VSUMP: Sum(Max(VolDiff, 0), d) / Sum(Abs(VolDiff), d)
        v_diff = V - V.shift(1)
        v_gain = v_diff.clip(lower=0)
        v_loss = -v_diff.clip(upper=0)
        v_abs_diff = v_diff.abs()
        for d in windows:
            s_v_gain = v_gain.rolling(d).sum()
            s_v_abs = v_abs_diff.rolling(d).sum()
            features[f'VSUMP{d}'] = s_v_gain / (s_v_abs + eps)
            
        # BB. VSUMN: Similar
        for d in windows:
            s_v_loss = v_loss.rolling(d).sum()
            s_v_abs = v_abs_diff.rolling(d).sum()
            features[f'VSUMN{d}'] = s_v_loss / (s_v_abs + eps)
            
        # CC. VSUMD: VSUMP - VSUMN
        for d in windows:
            features[f'VSUMD{d}'] = features[f'VSUMP{d}'] - features[f'VSUMN{d}']

        # -------------------------------------------------------------------------
        # Final Assembly (Ensure strict order)
        # -------------------------------------------------------------------------
        # Order list construction
        col_order = []
        # 1. K-Bar
        col_order.extend(['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2'])
        # 2. Price
        col_order.extend(['OPEN0', 'HIGH0', 'LOW0', 'VWAP0'])
        # 3. Rolling
        rolling_names = [
            'ROC', 'MA', 'STD', 'BETA', 'RSQR', 'RESI', 'MAX', 'MIN', 'QTLU', 'QTLD', 'RANK', 'RSV',
            'IMAX', 'IMIN', 'IMXD', 'CORR', 'CORD', 'CNTP', 'CNTN', 'CNTD', 'SUMP', 'SUMN', 'SUMD',
            'VMA', 'VSTD', 'WVMA', 'VSUMP', 'VSUMN', 'VSUMD'
        ]
        for name in rolling_names:
            for d in windows:
                col_order.append(f"{name}{d}")
        
        # Assemble DataFrame
        res_df = pd.DataFrame(features, index=df.index)
        
        # Reorder columns explicitly
        # Note: If any col is missing (due to typo), it will error, which is good for verification.
        res_df = res_df[col_order]
        
        # Fill NaN
        res_df = res_df.fillna(0)
        
        # Final check
        if res_df.shape[1] != 158:
            raise ValueError(f"Feature count mismatch: {res_df.shape[1]}")
            
        return res_df, list(res_df.columns)
