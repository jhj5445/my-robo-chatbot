import pandas as pd
import yfinance as yf

def calculate_portfolio_performance_from_history(records):
    """
    history records: list of dicts.
    Calculates INDEPENDENT performance for each record (Cohort Analysis).
    Returns: DataFrame with columns ['Date', 'Port_2024-01-15', 'Port_2024-01-16'...], ErrorMsg
    """
    if not records:
        return None, "기록이 없습니다."
        
    # 1. Identify Time Range and Universe (Union of all)
    all_tickers = set()
    start_date_min = None
    
    for rec in records:
        if 'weights' in rec:
            all_tickers.update(rec['weights'].keys())
        else:
            all_tickers.update(rec['items'])
            
        r_date = pd.to_datetime(rec['date'])
        if start_date_min is None or r_date < start_date_min:
            start_date_min = r_date
            
    if not all_tickers:
        return None, "종목 정보가 없습니다."

    # 2. Download Data (Batch)
    end_date = pd.Timestamp.now().normalize()
    try:
        # Buffer
        s_dt = start_date_min
        data = yf.download(list(all_tickers), start=s_dt, end=end_date + pd.Timedelta(days=1), progress=False)
        
        # Handle Data Structure
        if isinstance(data.columns, pd.MultiIndex):
            if 'Adj Close' in data.columns.levels[0]: df_close = data['Adj Close']
            elif 'Close' in data.columns.levels[0]: df_close = data['Close']
            else: return None, "가격 데이터 불가"
        else:
             if 'Adj Close' in data.columns: df_close = data[['Adj Close']]
             elif 'Close' in data.columns: df_close = data[['Close']]
             else: return None, "데이터 형식 호환 불가"
        
        df_close = df_close.ffill().bfill()
        # Drop Timezone if present to match record dates
        if df_close.index.tz is not None:
             df_close.index = df_close.index.tz_localize(None)
             
    except Exception as e:
        return None, f"데이터 다운로드 실패: {e}"

    # 3. Calculate Each Cohort Independent Return
    combined_df = pd.DataFrame(index=df_close.index)
    # Filter valid range
    combined_df = combined_df[combined_df.index >= start_date_min]
    
    cohort_cols = []
    
    for rec in records:
        rec_date_str = rec.get('date')
        rec_lbl = f"Port_{rec_date_str}"
        
        rec_dt = pd.to_datetime(rec_date_str)
        
        # Get Weights
        w_dict = rec.get('weights', {})
        if not w_dict:
            items = rec.get('items', [])
            if items:
                w = 1.0 / len(items)
                w_dict = {t: w for t in items}
        
        if not w_dict: continue
        
        # Calculate daily weighted return starting from rec_dt
        # We start tracking from Day T (rec_dt) Close (assuming buy on close) or T+1?
        # Usually Backtest: Buy on T Close. Return starts T -> T+1.
        # So base value 1.0 is at T. T+1 has value 1.0 * (1+ret).
        
        # Slice price data from rec_dt
        sub_prices = df_close[df_close.index >= rec_dt].copy()
        if sub_prices.empty: continue
        
        # Normalize weights
        total_w = sum(w_dict.values())
        norm = 100.0 if total_w > 1.5 else 1.0
        
        # Calculate Basket Value (Vectorized)
        # Value_t = Sum(Shares_i * Price_i_t)
        # But we have Weights at T0.
        # Shares_i = (Initial_Capital * w_i) / Price_i_T0
        # Value_t = Sum(Shares_i * Price_i_t) / Initial_Capital
        # Value_t = Sum( (w_i/Price_i_T0) * Price_i_t )
        
        # We need Price_i_T0 (First row of sub_prices)
        base_prices = sub_prices.iloc[0]
        
        # Calculate generic curve
        curve = pd.Series(0.0, index=sub_prices.index)
        
        valid_tickers = 0
        for t, w in w_dict.items():
            if t in sub_prices.columns:
                p0 = base_prices[t]
                if pd.isna(p0) or p0 == 0: continue
                # Contribution = w * (Pt / P0)
                # If w is 10%, contribution starts at 0.1.
                # Sum of contributions at t=0 must be 1.0 (if fully invested)
                contrib = (w / norm) * (sub_prices[t] / p0)
                curve += contrib
                valid_tickers += 1
        
        if valid_tickers > 0:
            combined_df[rec_lbl] = curve
            cohort_cols.append(rec_lbl)
            
    # Clean up: Drop rows where all cohorts are NaN (before any start)
    # Actually explicit start_date_min handling is better.
    
    return combined_df[cohort_cols], None

