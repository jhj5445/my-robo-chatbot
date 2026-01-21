import streamlit as st
import google.generativeai as genai
import ssl
import requests
import warnings
from io import StringIO
import yfinance as yf

# -----------------------------------------------------------------------------
# SSL Fix for FinanceDataReader & KRX (User Environment Specific)
# -----------------------------------------------------------------------------
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Requests Verify Patch
from requests.packages.urllib3.exceptions import InsecureRequestWarning
warnings.simplefilter('ignore', InsecureRequestWarning)

# Monkey-patch requests to disable verification by default
if not hasattr(requests, '_original_get'):
    requests._original_get = requests.get
    requests._original_post = requests.post

    def new_get(*args, **kwargs):
        kwargs['verify'] = False
        return requests._original_get(*args, **kwargs)

    def new_post(*args, **kwargs):
        kwargs['verify'] = False
        return requests._original_post(*args, **kwargs)

    requests.get = new_get
    requests.post = new_post
# -----------------------------------------------------------------------------

import os
import glob
import re
import streamlit.components.v1 as components
import yfinance as yf
import plotly.express as px
import pandas as pd
import pickle # Added for Persistence
import datetime
import json
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# -----------------------------------------------------------------------------
# Helper Functions (Moved to Top for Scope Safety)
# -----------------------------------------------------------------------------
import pickle
try:
    import stats_helper # Custom Helper
except ImportError:
    pass 

try:
    from alpha_provider import AlphaFactory
except Exception as e:
    import streamlit as st
    st.error(f"❌ Critical Error importing alpha_provider: {e}")
    print(f"❌ Critical Error importing alpha_provider: {e}")
    AlphaFactory = None

MODEL_SAVE_DIR = "saved_models"
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

def save_model_checkpoint(model_name, data):
    try:
        if not os.path.exists(MODEL_SAVE_DIR):
            os.makedirs(MODEL_SAVE_DIR)
        
        filepath = os.path.join(MODEL_SAVE_DIR, f"{model_name}.pkl")
        filepath = filepath.replace(":", "-").replace("|", "_")
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def load_model_checkpoint(model_name):
    try:
        filepath = os.path.join(MODEL_SAVE_DIR, f"{model_name}.pkl")
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
    return None

def calculate_feature_set(df, feature_level):
    df = df.copy()
    feature_cols = []

    # 0. Alpha158 (Qlib Exact Match)
    if "Alpha158" in feature_level:
        if AlphaFactory:
            # [Fix] Unpack tuple (df, columns)
            alpha_df, _ = AlphaFactory.get_alpha158(df)
            
            # Merge alphas back to df
            # alpha_df index is MultiIndex (Date, Ticker) or matches df index
            # If df is single ticker, alpha_df might have MultiIndex.
            # Align indices
            if isinstance(alpha_df.index, pd.MultiIndex) and not isinstance(df.index, pd.MultiIndex):
                 # Drop ticker level if single ticker
                 alpha_df = alpha_df.reset_index(level=1, drop=True)
            
            # Join columns
            df = df.join(alpha_df, rsuffix='_alpha') # safety
            feature_cols = alpha_df.columns.tolist()
            return df, feature_cols
        else:
            print("AlphaFactory not found, falling back to Rich")

    # 1. Light (Basic 5)
    if "Light" in feature_level:
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['Disparity_5'] = df['Close'] / df['MA5']
        df['Disparity_20'] = df['Close'] / df['MA20']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['Volatility'] = df['Close'].pct_change().rolling(20).std()
        df['Momentum_1M'] = df['Close'].pct_change(20)
        
        feature_cols = ['Disparity_5', 'Disparity_20', 'RSI', 'Volatility', 'Momentum_1M']
    else:
        # Standard(22) or Rich(50+)
        if "Rich" in feature_level:
            windows = [3, 5, 10, 20, 40, 60, 120]
        else:
            windows = [5, 10, 20, 60]

        df['Ret_1d'] = df['Close'].pct_change()
        
        for w in windows:
            col_roc = f'ROC_{w}'
            df[col_roc] = df['Close'].pct_change(w)
            feature_cols.append(col_roc)
            
            col_ma = f'MA_Dist_{w}'
            ma = df['Close'].rolling(window=w).mean()
            df[col_ma] = df['Close'] / ma
            feature_cols.append(col_ma)
            
            col_vol = f'Vol_{w}'
            df[col_vol] = df['Ret_1d'].rolling(window=w).std()
            feature_cols.append(col_vol)
            
            col_vol_ratio = f'Vol_Ratio_{w}'
            vol_ma = df['Volume'].rolling(window=w).mean()
            df[col_vol_ratio] = df['Volume'] / vol_ma
            feature_cols.append(col_vol_ratio)
        
        # RSI
        rsi_windows = [9, 14, 28, 60] if "Rich" in feature_level else [14, 60]
        for rsi_w in rsi_windows:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(rsi_w).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(rsi_w).mean()
            rs = gain / loss
            col_rsi = f'RSI_{rsi_w}'
            df[col_rsi] = 100 - (100 / (1 + rs))
            feature_cols.append(col_rsi)

        # [Rich Only Features]
        if "Rich" in feature_level:
            # Lagged Returns
            for lag in [1, 2, 3, 5]:
                col_lag = f'Ret_Lag_{lag}'
                df[col_lag] = df['Ret_1d'].shift(lag)
                feature_cols.append(col_lag)
            
            # Candle Patterns
            df['Candle_Body'] = (df['Close'] - df['Open']).abs()
            df['Candle_Len'] = (df['High'] - df['Low'])
            df['Body_Ratio'] = df['Candle_Body'] / df['Candle_Len'].replace(0, 1)
            feature_cols.append('Body_Ratio')
            
            df['Shadow_Upper'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Candle_Len'].replace(0, 1)
            df['Shadow_Lower'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Candle_Len'].replace(0, 1)
            feature_cols.append('Shadow_Upper')
            feature_cols.append('Shadow_Lower')
            
            # Day of Week
            df['DayOfWeek'] = df.index.dayofweek
            feature_cols.append('DayOfWeek')
            
    feature_cols = list(set(feature_cols)) # Ensure unique
    return df, feature_cols

# -----------------------------------------------------------------------------
# Portfolio & Universe Helpers (Moved to Top for Scope Safety)
# -----------------------------------------------------------------------------
PORT_SAVE_DIR = "saved_portfolios"
if not os.path.exists(PORT_SAVE_DIR):
    os.makedirs(PORT_SAVE_DIR)

PORTFOLIO_HISTORY_FILE = os.path.join(PORT_SAVE_DIR, "ai_portfolio_history.json")

def load_portfolio_history():
    if os.path.exists(PORTFOLIO_HISTORY_FILE):
        try:
            with open(PORTFOLIO_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_portfolio_history(history_data):
    try:
        with open(PORTFOLIO_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving portfolio history: {e}")

# NASDAQ 100 Full List (Static)
NASDAQ_100_FULL = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ADBE", "COST",
    "PEP", "CSCO", "NFLX", "AMD", "TMUS", "INTC", "TXN", "QCOM", "AMGN", "HON",
    "AMAT", "INTU", "SBUX", "ADP", "BKNG", "GILD", "ISRG", "MDLZ", "REGN", "VRTX",
    "LRCX", "ADI", "PANW", "MU", "SNPS", "CDNS", "CHTR", "KLAC", "CSX", "MAR",
    "CRWD", "MELI", "NXPI", "ORLY", "CTAS", "MNST", "ROP", "LULU", "ODFL", "PCAR",
    "PAYX", "FTNT", "KDP", "EXC", "XEL", "IDXX", "BIIB", "AEP", "MCHP", "ALGN",
    "DLTR", "EA", "AZN", "WBD", "FAST", "CTSH", "BKR", "GFS", "VRSK", "KHC",
    "GEHC", "TEAM", "SGEN", "ZS", "DDOG", "FANG", "ON", "ANSS", "CDW", "TTD",
    "WBA", "ILMN", "SIRI", "ZM", "ENPH", "JD", "PDD", "BIDU", "NTES", "CEG",
    "FISV", "ATVI", "MRVL", "MRNA", "DXCM", "LCID", "RIVN", "WDAY", "EBAY", "SPLK"
]
import lightgbm as lgb
import numpy as np
import numpy as np
import scipy.optimize as sco
from pykrx import stock
import time
from datetime import datetime, timedelta


# 1. API 키 설정 (Google AI Studio에서 발급받은 키 입력)
# 안전한 방식 (Streamlit Secrets 사용)
# -----------------------------------------------------------------------------
# 1. API 키 설정 (Rotation Logic)
# -----------------------------------------------------------------------------
# 사용 가능한 모든 API 키를 로드합니다.
api_keys = []

# 1. Streamlit Secrets 우선 확인
if "GOOGLE_API_KEY" in st.secrets:
    api_keys.append(st.secrets["GOOGLE_API_KEY"])
    # 추가 키 확인 (GOOGLE_API_KEY_2, _3 ...)
    i = 2
    while f"GOOGLE_API_KEY_{i}" in st.secrets:
        api_keys.append(st.secrets[f"GOOGLE_API_KEY_{i}"])
        i += 1
else:
    # 2. 환경 변수 확인 (로컬 테스트)
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        api_keys.append(key)
        # 추가 키 확인
        i = 2
        while os.getenv(f"GOOGLE_API_KEY_{i}"):
            api_keys.append(os.getenv(f"GOOGLE_API_KEY_{i}"))
            i += 1

if not api_keys:
    st.error("API 키가 설정되지 않았습니다. .streamlit/secrets.toml 또는 환경변수를 확인해주세요.")
    st.stop()

# 기본 키로 초기 설정
genai.configure(api_key=api_keys[0])

def generate_content_with_rotation(prompt, model_name="gemini-3-flash-preview"):
    """
    API 키를 순환하며 컨텐츠 생성을 시도합니다.
    Rate Limit 발생 시 다음 키로 자동 전환합니다.
    """
    last_error = None
    
    for i, key in enumerate(api_keys):
        try:
            # 현재 키로 설정 및 생성 시도
            genai.configure(api_key=key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            last_error = e
            # Quota 관련 에러인 경우 다음 키 시도
            # (429 Resource exhausted, Quota exceeded 등)
            error_str = str(e)
            if "429" in error_str or "Quota" in error_str or "Resource exhausted" in error_str:
                # 마지막 키가 아니면 다음 키로 계속
                if i < len(api_keys) - 1:
                    continue
            
            # 그 외 에러거나 마지막 키면 루프 종료 (After loop raises)
            break
            
    # 모든 시도 실패 시 에러 발생
    raise last_error

# -----------------------------------------------------------------------------

# 2. FAQ 데이터 정의 (여기에 준비하신 FAQ 내용을 자유롭게 넣으세요)
faq_data = """
[로보어드바이저 서비스 상세 매뉴얼]

### 1. 서비스 기능 및 기본 원칙
- **가입 및 설계**: 가입과 동시에 맞춤설계가 진행되는 것이 아니며, 고객이 가입 후 직접 맞춤설계를 진행해야 합니다. 이는 고객에게 포트폴리오를 '추천'드리는 서비스이기 때문입니다.
- **투자성향**: 투자자성향과 상관없이 가입은 가능하며, 최종에는 본인 투자성향에 따른 포트폴리오 유형이 선택되지만 타 유형도 선택 가능합니다. 단, 맞춤설계 과정에서 투자부적합성 안내, 약관동의 등 필수 고지사항 프로세스가 추가 발생합니다.
- **포트폴리오 비중 수정**: 투자자가 임의로 자산군 비중을 수정하거나 일부 펀드만 교체하게 설정할 수 없습니다. 퇴직연금의 위험자산비율 준수를 위해 로보어드바이저가 매매 시 자동으로 비중 준수를 위한 매매를 진행합니다.
- **펀드 교체 범위**: 원칙적으로 고객이 가지고 계신 공모펀드 전체가 교체 대상입니다. 단, 로보어드바이저가 판단하기에 1위 상품과 성능이 유사한 펀드는 일부만 매매될 수 있으며, 거래 범주에서 제외되는 항목은 매매에서 제외됩니다.

### 2. 가입 불가 요건 상세 (각 계좌별)
- **퇴직연금**: MP구독 서비스 이용 계좌 등.
- **개인연금**: 연금개시 정기지급 계좌(임의식은 가능), 대출 약정계좌, 연금저축계좌 정기매도 약정계좌, 자동대체입금매수 약정계좌, CMS자동대체입금매수 약정계좌, 펀드 정기자동매수 약정계좌, 이전용 계좌, 이관신청중 계좌, 사고계좌(매매제한) 및 장기미사용 계좌. 타 서비스 이용 중인 경우(개인연금 랩, 개인연금 자문, 적립식 자동매수 서비스-연금 모으기) 가입 불가.
- **ISA**: 계좌해지 신청중 및 이후단계 계좌, 자동대체입금매수 약정계좌, CMS자동대체입금매수 약정계좌, 펀드 정기자동매수 약정계좌, 사고계좌(매매제한), 적립식 자동매수 서비스, 이관신청접수/이관해지신청 계좌, 만기초과 계좌.
- **일반계좌**: 계좌해지 신청중 및 이후단계 계좌, 자동대체입금매수 약정계좌, CMS자동대체입금매수 약정계좌, 사고계좌(매매제한), 적립식 자동매수 서비스, 이관신청접수/이관해지신청 계좌, 만기초과 계좌, 신용/대출/대여/제휴 약정계좌, 공모부동산분리과세 약정, 분리과세하이일드 약정, 분리과세고위험고수익펀드 약정, 월지급 약정, 계좌증거금률 100% 외, 해외주식 계좌증거금률 100% 외, 계좌위탁증거금 미징수, 매매허용 유가증권 '펀드' 미등록 계좌. 랩계약 약정계좌, 자문사 일임/자문 계좌 이용 시 불가.
- **비과세종합저축**: 위 일반계좌 요건과 동일함.

### 3. 맞춤설계 이용 제한 여부 및 제외 상품 (MAPIS 기준)
- **정상 상태 기준 (MAPIS 7895, 8525)**: 
  * 투자자성향: 성장형, 성장추구형 등 (안정추구형 등 부적합 시 '해당'으로 표시됨)
  * 투자권유: '희망' 상태여야 함
  * 운용가능금액: 10,000원 이상
  * 위험자산비율: 퇴직연금의 경우 70% 이하
  * 보유상품갯수: 퇴직연금 20개 미만, 개인연금/ISA 등 50개 이하
- **운용 및 평가 제외 펀드 리스트**: 아래 펀드는 펀드평가금액 및 운용가능금액 집계에서 제외됩니다.
  1) 거래불가펀드 (예: 러시아 펀드 등)
  2) 환매수수료 발생 펀드
  3) 사모펀드
  4) 오프라인전용펀드
  5) 환매금지펀드
  6) 성과보수 펀드
  7) 코스닥벤처 펀드
  8) 숙려대상 펀드

### 4. 2026/1/1 최신 제한 및 업데이트 사항
- **현재자금투자성향 제한**: Q1 '단기 생계 자금' 혹은 Q2 '원금 보존 추구' 답변 중 하나라도 체크된 경우 이용 불가. (MAPIS 3250에서 확인 및 재진단 필요)
- **ISA 해지 관련**: ISA 부적합 요건 발생 혹은 계좌 이전 신청 시, 해당 계좌의 로보어드바이저 가입을 사전적으로 해지해야 업무 진행이 가능함.
- **투자설명서 발송 (25/10/24 추가)**:
  * 퇴직연금: 맞춤설계 완료 직후 1회만 알림톡 발송.
  * 개인연금/ISA/일반: 매수되는 펀드마다 매수 시점에 개별 투자설명서 발송.

### 5. 매매, 수익률 및 알림 규칙
- **매매 불가 시간**: 23시 55분 ~ 24시 05분 (주문 제출 시 실패 및 전체 취소 처리).
- **리밸런싱 알림**: 수시(직전 승인 5영업일 경과 및 비중 차이 발생 시, 약 14일 주기 검출), 정기(최종 승인 후 40영업일 경과 시).
- **수익률 확인 불가 사유**: 계좌 내 로보 미운용 상품(예금, ELB, ETF 등) 존재로 인한 혼동 방지 및 고객 의사(승인/거절) 개입에 따른 성과 차이 때문.
- **성과 확인 경로**: [MY 로보어드바이저 > 계좌현황 > 보유펀드] 또는 [MY 펀드] 화면.

### 6. 주요 에러 사례 (Error Case)
1) **소수점 처리**: 퇴직연금에서 아주 적은 금액 투자 시 비중 단위가 정수이기 때문에 '매도 상품 없음' 에러 발생 가능.
2) **위험자산 비중 시차**: 당일 매매로 위험자산비율 70% 초과 상태에서 설계 시 장애 발생. 결제 완료 시까지 대기 필요.
3) **중복 프로세스**: '포트폴리오 변경이 진행 중입니다' 팝업 시 기존 매매 스케줄 취소 후 재진행.
4) **미국 국적자**: 매매 불가 펀드 포함 시 맞춤설계 단계에서 에러.
5) **퇴사 후 DC 계좌**: 가입자 번호가 남아 화면 진입은 가능하나 설계 시 에러 발생 (차세대 이후 수정 예정).
"""

# 3. 모델 설정 (Gemini 3 Flash 사용)
# 시스템 프롬프트에 FAQ 데이터를 주입합니다.
system_prompt = f"""
답변의 1순위 근거는 제공된 **[FAQ 데이터]**입니다.
만약 FAQ에 없는 내용 중 일반적인 금융 지식은 당신의 기본 지식을 활용해 설명하되, 미래에셋의 구체적인 수치나 정책은 추측하지 마세요.
민감한 투자 권유 질문에는 FAQ의 공식 입장을 전달하세요.
일반적인 지식이 아니고, [FAQ 데이터]에 없는 내용이라면 "죄송합니다. 해당 내용은 고객센터(1588-XXXX)로 문의 부탁드립니다"라고 답변하세요."
[FAQ 데이터]
{faq_data}
"""

model = genai.GenerativeModel(
    model_name="gemini-3-flash-preview",
    system_instruction=system_prompt
)

# 4. 웹 화면 UI 구성 (Streamlit)
st.set_page_config(page_title="로보어드바이저 챗봇", page_icon="🤖", layout="wide")

# OP.GG 스타일 커스텀 CSS 적용 (Light Theme)
st.markdown(
    """
    <style>
        /* 기본 폰트 설정 */
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Noto Sans KR', sans-serif;
        }

        /* 메인 배경색 - OP.GG의 밝은 회색/블루 톤 */
        .stApp {
            background-color: #ecf2f5;
            color: #23292f; /* 짙은 회색 텍스트 */
        }

        /* 사이드바 배경색 - OP.GG의 짙은 네이비 (헤더 느낌) */
        [data-testid="stSidebar"] {
            background-color: #1c2836;
        }
        
        /* 사이드바 내 텍스트 색상 조정 (더 구체적으로) */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
            color: #ffffff !important;
        }

        /* 라디오 버튼 스타일 커스텀 (사이드바 메뉴) */
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            background-color: transparent;
            color: #b0b8c1 !important; /* 기본: 회색 */
            padding: 10px;
            border-radius: 4px;
            transition: all 0.2s;
            margin-bottom: 2px;
            cursor: pointer;
        }
        
        /* 라디오 버튼 선택된 항목 */
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] {
             background-color: #5383e8 !important; /* 선택시 블루 배경 */
             color: #ffffff !important; /* 선택시 흰글씨 */
             font-weight: bold;
        }
        
        /* 라디오 버튼 호버 효과 */
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
             background-color: #24354a; /* 호버시 약간 밝은 네이비 */
             color: #ffffff !important;
        }

        /* 헤더 배경색 */
        [data-testid="stHeader"] {
            background-color: rgba(0,0,0,0);
        }

        /* 제목 색상 (OP.GG 브랜드 블루 포인트) */
        h1 {
            color: #5383e8 !important;
            font-weight: 700;
        }
        h2, h3 {
            color: #23292f !important;
        }

        /* 채팅 입력창 스타일 (화이트 박스) */
        div[data-testid="stChatInput"] > div {
            background-color: #ffffff !important;
            border: 1px solid #dce2f0 !important;
            border-radius: 4px; /* 살짝 덜 둥글게 */
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        /* 입력창 텍스트 영역 */
        div[data-testid="stChatInput"] textarea {
            background-color: transparent !important;
            color: #23292f !important; /* 어두운 글씨 */
        }
        
        /* 플레이스홀더 텍스트 */
        div[data-testid="stChatInput"] textarea::placeholder {
            color: #9aa4af !important;
        }

        /* 포커스 효과 (브랜드 블루) */
        div[data-testid="stChatInput"] > div:focus-within {
            border-color: #5383e8 !important;
            box-shadow: 0 0 0 1px #5383e8 !important;
        }

        /* 버튼 스타일 (브랜드 블루) */
        .stButton button {
            background-color: #5383e8;
            color: white;
            border: none;
            border-radius: 4px;
            font-weight: bold;
        }
        .stButton button:hover {
            background-color: #426cb7;
            color: white;
        }

        /* 메시지 박스 스타일 (채팅 풍선 느낌) */
        .stChatMessage {
            background-color: transparent;
        }
        
        /* 사용자/AI 메시지 구분감 (선택 사항) */
        [data-testid="chatAvatarIcon-user"] {
            background-color: #5383e8;
        }
        [data-testid="chatAvatarIcon-assistant"] {
            background-color: #ffb900; /* AI는 노란색 포인트 */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 사이드바 네비게이션
with st.sidebar:
    st.title("🚀 Antigravity")
    
    selection = st.radio(
        "메뉴 선택", 
        ["🏠 홈 (Dashboard)", "📄 Macro Talking Point", "🤖 로보 어드바이저 (Demo)", "🔎 ETF 구성 종목 검색", "🤖 AI 모델 테스팅", "🧪 Qlib 실험실 (Pro)", "🔍 기술적 패턴 스캐너", "⚖️ 포트폴리오 최적화", "🤖 챗봇"]
    )

import requests

# -----------------------------------------------------------------------------
# Helper Functions for Ticker Fetching
# -----------------------------------------------------------------------------
@st.cache_data
def get_sp500_tickers():
    """Wikipedia에서 S&P 500 종목 리스트를 가져옵니다."""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        tables = pd.read_html(response.text)
        df = tables[0]
        tickers = df['Symbol'].tolist()
        return [t.replace('.', '-') for t in tickers] # BRK.B -> BRK-B 변환
    except Exception as e:
        st.error(f"S&P 500 리스트를 가져오는 중 오류: {e}")
        return []

@st.cache_data
def get_nasdaq100_tickers():
    """Wikipedia에서 NASDAQ 100 종목 리스트를 가져옵니다."""
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        tables = pd.read_html(response.text)
        # 테이블 인덱스가 바뀔 수 있으므로 열 이름으로 확인
        for table in tables:
            if 'Ticker' in table.columns:
                return [t.replace('.', '-') for t in table['Ticker'].tolist()]
            elif 'Symbol' in table.columns:
                return [t.replace('.', '-') for t in table['Symbol'].tolist()]
        return []
    except Exception as e:
        st.error(f"NASDAQ 100 리스트를 가져오는 중 오류: {e}")
        return []



if selection == "🏠 홈 (Dashboard)":
    st.title("🏠 Antigravity Dashboard")
    st.markdown("### 🚀 AI가 분석하는 실전 트레이딩/투자 플랫폼")
    
    # Simple Dashboard Widgets
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("S&P 500 (SPY)", "Live", "checking...")
    with col2:
        st.metric("NASDAQ 100 (QQQ)", "Live", "checking...")
    with col3:
        st.metric("KOSPI 200", "Live", "checking...")
        
    st.divider()
    
    st.info("👈 **사이드바에서 원하는 분석 도구를 선택하세요.**")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🤖 주요 기능")
        st.markdown("""
        - **🤖 로보 어드바이저**: 내 성향에 맞는 ETF 포트폴리오
        - **🔎 ETF 역검색**: 내가 원하는 종목을 담은 ETF 찾기
        - **🧪 Qlib 실험실**: Microsoft Qlib 기반 AI 알파 모델링
        """)
        
    with c2:
        st.markdown("#### 📢 Market Briefing")
        if st.button("✨ 오늘의 시황 브리핑 생성 (Gemini)"):
            with st.spinner("뉴스와 시세를 분석 중입니다..."):
                try:
                    prompt = "오늘의 미국 증시 및 한국 증시 주요 이슈를 3줄로 요약해줘."
                    summary = generate_content_with_rotation(prompt)
                    st.success(summary)
                except Exception as e:
                    st.error(f"생성 실패: {e}")

if selection == "🤖 챗봇":
    st.title("🤖 로보어드바이저 상담")
    st.caption("FAQ 데이터를 기반으로 AI가 답변해 드립니다.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 기존 대화 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 추천 질문 (FAQ) 영역 - 대화 기록 아래에 배치
    # 명확한 키워드로 직접 정의
    faq_keywords = [
        "서비스 가입/설계",
        "포트폴리오 비중 수정",
        "퇴직연금 가입제한",
        "개인연금 가입제한",
        "매매/리밸런싱 규칙",
        "수익률 미노출 사유",
        "주요 에러 사례"
    ]

    with st.expander("💡 자주 묻는 질문 (추천 키워드)"):
        st.caption("궁금한 내용을 클릭해보세요.")
        cols = st.columns(4) # 4열로 배치
        for i, keyword in enumerate(faq_keywords):
            if cols[i % 4].button(keyword, key=f"faq_{i}"):
                st.session_state.messages.append({"role": "user", "content": f"{keyword}에 대해 알려줘"})
                st.rerun()
            
    # 가장 최근 메시지가 user이고 assistant의 답변이 없을 때 (버튼 클릭 직후) 답변 생성 트리거
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        # 이미 답변이 달린 적이 있는지 확인 (마지막이 user면 답변해야 함)
        # 하지만 Streamlit 구조상 chat_input 루프 밖에서 처리해야 자연스러움.
        # 여기서는 chat_input이 아래에 있어서, 버튼 클릭 -> rerun -> 여기까지 옴 -> 
        # 화면에 user msg 표시됨 -> 이제 assistant msg 표시할 차례.
        
        # 마지막 메시지가 assistant가 아닐 경우에만 답변 생성 시도
        # (주의: chat_input을 통한 입력은 아래 블록에서 처리되므로, 여기서는 버튼 클릭으로 인한 경우만 처리하면 좋음.
        #  그러나 간단하게직전 메시지가 user면 무조건 답변하게 로직을 통합하는게 깔끔함.
        #  다만 아래 chat_input 로직과 중복되지 않게 해야 함.)
        pass 

    # 사용자 질문 입력
    if prompt := st.chat_input("궁금한 점을 입력하세요"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    # 답변 생성 로직 (버튼 클릭 or 입력창 입력 공통 처리)
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            try:
                # 마지막 사용자 메시지 가져오기
                last_user_msg = st.session_state.messages[-1]["content"]
                # API Key Rotation 적용
                full_prompt = f"질문: {last_user_msg}\n\n답변 (한국어로, 금융 전문가처럼):"
                response_text = generate_content_with_rotation(full_prompt)
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

if selection == "📄 Macro Talking Point":
    st.title("📄 Macro Talking Point")
    st.caption("각 지수와 날짜별 리포트를 확인하세요.")
    
    # 비밀번호 보호 기능 추가
    input_password = st.text_input("접근 암호를 입력하세요", type="password")
    
    # Secrets에서 설정한 비밀번호와 비교
    correct_password = st.secrets["MACRO_PAGE_PASSWORD"]
    
    if input_password != correct_password:
        st.warning("🔒 올바른 암호를 입력해야 내용을 볼 수 있습니다.")
        st.stop()
        
    st.success("🔓 인증되었습니다.")
    
    # -----------------------------------------------------------------------------
    # HTML 리포트 뷰어 (Iframe 방식)
    # -----------------------------------------------------------------------------
    # 지정된 디렉토리 내의 모든 HTML 파일을 찾아서 목록으로 보여줍니다.
    # NOTE: The original code used glob.glob and a specific naming convention.
    # The provided snippet suggests a different directory and naming convention.
    # For this edit, I will assume the user wants to keep the original report loading
    # mechanism but add the password protection.
    # If the user intended to replace the report loading logic, a separate instruction
    # would be needed.
    
    # 리포트 파일 스캔 함수
    def get_reports():
        # 현재 디렉토리의 html 파일 검색
        files = glob.glob("Macro Talking Point_ *.html")
        reports = []
        for f in files:
            # 파일명 파싱: "Macro Talking Point_ {Index}_{Date}.html"
            # 예: "Macro Talking Point_ CPI_20251216.html"
            match = re.search(r"Macro Talking Point_ (.+?)_(\d+)\.html", f)
            if match:
                index_name = match.group(1)
                date_str = match.group(2)
                reports.append({
                    "filename": f,
                    "index": index_name,
                    "date": date_str,
                    "display": f"[{date_str}] {index_name}"
                })
        
        # 날짜 내림차순 정렬
        reports.sort(key=lambda x: x["date"], reverse=True)
        return reports


    reports = get_reports()

    if not reports:
        st.warning("표시할 리포트 파일이 없습니다.")
    else:
        # 네비게이션(리포트 목록)을 사이드바에 배치하여 스크롤 시에도 고정되도록 변경
        with st.sidebar:
            st.divider() # 메뉴와 구분선
            st.markdown("### 📑 리포트 목록")
            
            # 1. 카테고리 필터링
            categories = sorted(list(set([r["index"] for r in reports])))
            categories.insert(0, "All")
            
            selected_category = st.selectbox("카테고리 선택:", categories)
            
            # 선택된 카테고리에 따라 리포트 필터링
            if selected_category == "All":
                filtered_reports = reports
            else:
                filtered_reports = [r for r in reports if r["index"] == selected_category]
            
            # 2. 리포트 선택
            if not filtered_reports:
                st.info("해당 카테고리에 리포트가 없습니다.")
                selected_report = None
            else:
                report_options = [r["display"] for r in filtered_reports]
                selected_option = st.radio("보고 싶은 리포트를 선택하세요:", report_options)
                
                # 선택된 리포트 정보 찾기
                selected_report = next((r for r in reports if r["display"] == selected_option), None)

        if selected_report:
            # 1. 스크롤 앵커 삽입 (이 위치로 스크롤을 땡겨올 예정)
            st.markdown('<div id="scroll-to-top-anchor"></div>', unsafe_allow_html=True)
            
            # 리포트 변경 시 스크롤을 맨 위로 초기화 (JS Injection)
            current_report_key = selected_report["filename"]
            if "last_viewed_report" not in st.session_state:
                st.session_state["last_viewed_report"] = None

            if st.session_state["last_viewed_report"] != current_report_key:
                st.session_state["last_viewed_report"] = current_report_key
                components.html(
                    f"""
                    <script>
                        // 리포트 키가 바뀔 때마다 실행: {current_report_key}
                        // 앵커(scroll-to-top-anchor)를 찾아서 scrollIntoView() 호출
                        // 렌더링 타이밍 문제를 피하기 위해 다중 시도 (Burst)
                        function forceScroll() {{
                            try {{
                                var anchor = window.parent.document.getElementById("scroll-to-top-anchor");
                                if (anchor) {{
                                    anchor.scrollIntoView({{behavior: 'auto', block: 'start'}});
                                }}
                            }} catch(e) {{}}
                        }}
                        
                        // 시도 1: 즉시
                        forceScroll(); 
                        // 시도 2: 0.3초 후 (DOM 렌더링 완료 예상)
                        setTimeout(forceScroll, 300);
                        // 시도 3: 0.8초 후 (혹시 늦게 로딩될 경우)
                        setTimeout(forceScroll, 800);
                    </script>
                    """,
                    height=0,
                    width=0
                )

            st.markdown(f"### 📑 {selected_report['index']} ({selected_report['date']})")
            
            try:
                with open(selected_report["filename"], "r", encoding="utf-8") as f:
                    html_content = f.read()
                
                # 높이 계산 로직 개선 (너무 길지 않게 튜닝)
                # HTML 태그들이 많으므로 라인 수 * 15px 정도로 축소 계산 (기존 25px -> 15px)
                line_count = len(html_content.splitlines())
                
                # 라인 수가 너무 적으면(minified) 기본 높이 부여, 아니면 라인 수 비례
                if line_count < 50:
                    estimated_height = 1200
                else:
                    estimated_height = max(800, line_count * 15 + 50)

                components.html(html_content, height=estimated_height, scrolling=True)
                
            except Exception as e:
                st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

if selection == "📈 전략 실험실 (Beta)":
    st.title("📈 나만의 주식 전략 실험실 (Beta)")
    st.caption("대표적인 투자 전략들을 내 입맛대로 설정해서 검증해보세요.")

    # 1. 설정 입력
    with st.expander("⚙️ 백테스팅 설정", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            ticker_input = st.text_input("종목 코드 (여러 개는 쉼표로 구분)", value="SPY, QQQ, AAPL")
        with col2:
            start_date = st.date_input("시작일", value=pd.to_datetime("2023-01-01"))
        with col3:
            end_date = st.date_input("종료일", value=pd.to_datetime("today"))

    st.divider()

    # 2. 전략 선택 및 파라미터 설정
    st.subheader("🛠️ 전략 구성하기")
    
    strategy_type = st.selectbox(
        "사용할 전략을 선택하세요.",
        ["이동평균선 크로스 (MA Crossover)", "RSI (상대강도지수)", "볼린저 밴드 (Bollinger Bands)"]
    )

    # 전략별 파라미터 UI (동적 변경)
    params = {}
    if strategy_type == "이동평균선 크로스 (MA Crossover)":
        st.info("💡 **골든크로스 전략**: 단기 이평선이 장기 이평선을 돌파하면 매수, 깨지면 매도합니다.")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            params['short_window'] = st.number_input("단기 이동평균 (일)", value=20, min_value=1)
        with col_p2:
            params['long_window'] = st.number_input("장기 이동평균 (일)", value=60, min_value=1)

    elif strategy_type == "RSI (상대강도지수)":
        st.info("💡 **RSI 역추세 전략**: 과매도 구간(매수 기준 미만)에서 매수하고, 과매수 구간(매도 기준 초과)에서 매도합니다.")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            params['window'] = st.number_input("RSI 기간", value=14, min_value=1)
        with col_p2:
            params['buy_threshold'] = st.number_input("매수 기준 (과매도)", value=30, min_value=0, max_value=100)
        with col_p3:
            params['sell_threshold'] = st.number_input("매도 기준 (과매수)", value=70, min_value=0, max_value=100)

    elif strategy_type == "볼린저 밴드 (Bollinger Bands)":
        st.info("💡 **볼린저 밴드 전략**: 주가가 하단 밴드를 터치하면 매수, 상단 밴드를 터치하면 매도합니다 (평균 회귀).")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            params['window'] = st.number_input("이동평균 기간", value=20, min_value=1)
        with col_p2:
            params['std_dev'] = st.number_input("표준편차 승수 (Standard Deviation multiplier)", value=2.0, step=0.1)

    # 3. 전략 실행 로직
    if st.button("🚀 전략 분석 실행"):
        with st.spinner("데이터 분석 및 전략 시뮬레이션 중..."):
            # 입력된 티커 파싱 (쉼표 구분)
            tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
            
            if not tickers:
                st.warning("종목 코드를 입력해주세요.")
                st.stop()
            
            results_list = []
            equity_curves = pd.DataFrame()
            
            # 진행상황바
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # SPY 벤치마크 데이터 다운로드 (시장 수익률 비교용)
            spy_total_return = 0.0
            try:
                spy_df = yf.download("SPY", start=start_date, end=end_date, progress=False)
                if not spy_df.empty:
                    if isinstance(spy_df.columns, pd.MultiIndex):
                        spy_df.columns = spy_df.columns.get_level_values(0)
                    
                    if 'Adj Close' not in spy_df.columns:
                         if 'Close' in spy_df.columns:
                            spy_df['Adj Close'] = spy_df['Close']
                    
                    if 'Adj Close' in spy_df.columns:
                        spy_return_series = spy_df['Adj Close'].pct_change()
                        spy_cum_return = (1 + spy_return_series).cumprod()
                        spy_total_return = spy_cum_return.iloc[-1] - 1
            except Exception as e:
                st.warning(f"SPY 벤치마크 데이터를 가져오는 데 실패했습니다: {e}")

            for i, ticker in enumerate(tickers):
                status_text.text(f"분석 중: {ticker} ({i+1}/{len(tickers)})")
                try:
                    # A. 데이터 다운로드
                    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    
                    if df.empty:
                        st.warning(f"'{ticker}': 데이터를 불러올 수 없습니다. 건너뜁니다.")
                        continue
                    
                    # yfinance 최신 버전 호환성
                    if isinstance(df.columns, pd.MultiIndex):
                        try:
                            df.columns = df.columns.get_level_values(0)
                        except:
                            pass

                    # 컬럼 이름 정리
                    if 'Adj Close' not in df.columns:
                        if 'Close' in df.columns:
                            df['Adj Close'] = df['Close']
                        else:
                            st.warning(f"'{ticker}': 가격 데이터 부족. 건너뜁니다.")
                            continue
                    
                    # 수익률 계산
                    df = df.copy() # 경고 방지
                    df['Return'] = df['Adj Close'].pct_change()
                    df.dropna(inplace=True)
                    
                    # B. 전략 로직 계산
                    df['Signal'] = 0 

                    # ---------------- [전략 함수 적용] ----------------
                    if strategy_type == "이동평균선 크로스 (MA Crossover)":
                        df['MA_Short'] = df['Adj Close'].rolling(window=params['short_window']).mean()
                        df['MA_Long'] = df['Adj Close'].rolling(window=params['long_window']).mean()
                        df.loc[df['MA_Short'] > df['MA_Long'], 'Signal'] = 1
                    
                    elif strategy_type == "RSI (상대강도지수)":
                        delta = df['Adj Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=params['window']).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=params['window']).mean()
                        rs = gain / loss
                        df['RSI'] = 100 - (100 / (1 + rs))
                        
                        import numpy as np
                        df['Signal'] = np.nan
                        df.loc[df['RSI'] < params['buy_threshold'], 'Signal'] = 1
                        df.loc[df['RSI'] > params['sell_threshold'], 'Signal'] = 0
                        df['Signal'] = df['Signal'].ffill().fillna(0)

                    elif strategy_type == "볼린저 밴드 (Bollinger Bands)":
                        df['MA'] = df['Adj Close'].rolling(window=params['window']).mean()
                        df['Std'] = df['Adj Close'].rolling(window=params['window']).std()
                        df['Upper'] = df['MA'] + (df['Std'] * params['std_dev'])
                        df['Lower'] = df['MA'] - (df['Std'] * params['std_dev'])
                        
                        import numpy as np
                        df['Signal'] = np.nan
                        df.loc[df['Adj Close'] < df['Lower'], 'Signal'] = 1
                        df.loc[df['Adj Close'] > df['Upper'], 'Signal'] = 0
                        df['Signal'] = df['Signal'].ffill().fillna(0)
                    # ------------------------------------------------

                    # C. 성과 계산
                    df['Strategy_Return'] = df['Signal'].shift(1) * df['Return']
                    df['Cumulative_Strategy'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()
                    df['Cumulative_Market'] = (1 + df['Return']).cumprod()
                    
                    # MDD
                    drawdown = df['Cumulative_Strategy'] / df['Cumulative_Strategy'].cummax() - 1
                    mdd = drawdown.min()
                    
                    # 최종 수익률
                    total_return = df['Cumulative_Strategy'].iloc[-1] - 1
                    market_return = df['Cumulative_Market'].iloc[-1] - 1 # Buy & Hold return
                    
                    # Alpha vs SPY
                    alpha_spy = total_return - spy_total_return

                    # 결과 저장
                    results_list.append({
                        "종목": ticker,
                        "전략 수익률": f"{total_return:.2%}",
                        "자체 B&H": f"{market_return:.2%}", # Buy and Hold
                        "SPY 수익률": f"{spy_total_return:.2%}",
                        "Alpha(vs SPY)": f"{alpha_spy:.2%}",
                        "MDD": f"{mdd:.2%}",
                        "Raw_Return": total_return # 정렬용
                    })
                    
                    # 차트용 데이터 (인덱스 통일)
                    equity_curves[ticker] = df['Cumulative_Strategy']
                
                except Exception as e:
                    st.warning(f"'{ticker}' 분석 중 오류: {e}")
                
                # 진행률 업데이트
                progress_bar.progress((i + 1) / len(tickers))
            
            status_text.empty()
            progress_bar.empty()

            if results_list:
                st.success(f"총 {len(results_list)}개 종목 분석 완료!")
                
                # 1. 요약 테이블 (수익률 순 정렬)
                results_df = pd.DataFrame(results_list)
                results_df = results_df.sort_values(by="Raw_Return", ascending=False).drop(columns=["Raw_Return"])
                
                st.subheader("📊 종목별 성과 (수익률 순)")
                st.caption(f"SPY(S&P 500) 수익률 ({start_date} ~ {end_date}): **{spy_total_return:.2%}**")
                st.dataframe(results_df, use_container_width=True)
                
                # 2. 비교 차트
                st.subheader("📈 수익률 비교 차트")
                if not equity_curves.empty:
                    # 인덱스(날짜)가 서로 다를 수 있으므로 fillna
                    equity_curves = equity_curves.fillna(method='ffill').fillna(1.0)
                    fig = px.line(equity_curves, title=f"전략 누적 수익률 비교 ({strategy_type})")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("분석된 결과가 없습니다. 티커를 확인해주세요.")

elif selection == "🤖 AI 모델 테스팅":
    st.title("🤖 AI 트레이딩 모델 연구소")
    st.caption("과거 데이터로 머신러닝 모델을 학습시켜 미래 수익률을 예측하고 검증합니다.")

    # Session State 초기화
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'gemini_insights' not in st.session_state:
        st.session_state.gemini_insights = {}

    # 1. 설정
    with st.expander("⚙️ 모델링 설정", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            universe_preset = st.selectbox(
                "분석 대상 유니버스", 
                ["직접 입력", "NASDAQ Top 10 (Demo)", "Tech Giants (M7)", "NASDAQ Top 30 (Big Tech)", "S&P 500 Top 50 (Sector Leaders)", "NASDAQ 100 + S&P 500 (Market Proxy)"]
            )
            
            if universe_preset == "직접 입력":
                tickers_input = st.text_input("종목 코드 입력 (쉼표 구분)", "AAPL, MSFT, GOOGL, AMZN, NVDA")
                tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
            
            elif universe_preset == "Tech Giants (M7)":
                tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
            
            elif universe_preset == "NASDAQ Top 10 (Demo)":
                tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "COST", "PEP"]

            elif universe_preset == "NASDAQ Top 30 (Big Tech)":
                # 시가총액 상위 등 주요 30개 종목
                tickers = [
                    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "COST", "PEP",
                    "CSCO", "NFLX", "AMD", "ADBE", "TMUS", "INTC", "QCOM", "TXN", "AMGN", "HON",
                    "AMAT", "INTU", "SBUX", "ADP", "BKNG", "GILD", "ISRG", "MDLZ", "REGN", "VRTX"
                ]
            
            elif universe_preset == "S&P 500 Top 50 (Sector Leaders)":
                # S&P 500 주요 종목 50개 (예시)
                tickers = [
                    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "LLY", "V",
                    "TSM", "UNH", "XOM", "JPM", "JNJ", "WMT", "MA", "PG", "HD", "AVGO", 
                    "CVX", "MRK", "ABBV", "COST", "PEP", "KO", "ADBE", "BAC", "CSCO", "CRM",
                    "MCD", "TMO", "ACN", "NFLX", "AMD", "LIN", "ABT", "DHR", "DIS", "NKE",
                    "WFC", "TXN", "NEE", "PM", "VZ", "RTX", "INTC", "QCOM", "UPS", "HON"
                ]

            elif universe_preset == "NASDAQ 100 + S&P 500 (Market Proxy)":
                # Option A: Market Proxy (Speed) vs Option B: Full Universe (Sloooow)
                scan_mode = st.radio("분석 모드 선택", ["🚀 속도 우선 (Top 100 Market Proxy)", "🐢 정밀 분석 (S&P 500 전종목 / ~5 min)"], horizontal=True)
                
                if "속도 우선" in scan_mode:
                    # Proxy for full market: NASDAQ 100 constituents (approx) + Key S&P 500
                    # 실시간으로 수천 개를 다 받으면 너무 느리므로, 대표 우량주 ~100개로 구성된 Proxy 사용
                    # 사용자가 요청한 '전체' 느낌을 내기 위해 섹터별 대표주를 최대한 많이 포함
                    tickers = [
                        # Tech
                        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ADBE", "CRM", "AMD", "INTC", "QCOM", "TXN", "IBM", "ORCL", "CSCO", "MU", "LRCX", "AMAT",
                        # Finance
                        "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "AXP", "V", "MA", "PYPL", "BRK-B", "SPGI",
                        # Health
                        "LLY", "UNH", "JNJ", "MRK", "ABBV", "PFE", "TMO", "ABT", "DHR", "BMY", "AMGN", "GILD", "ISRG", "VRTX", "REGN",
                        # Consumer
                        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "CL", "EL",
                        # Industrial / Energy / etc
                        "XOM", "CVX", "COP", "SLB", "EOG", "CAT", "DE", "HON", "GE", "LMT", "RTX", "BA", "UPS", "FDX", "UNP", "NEE", "DUK", "SO",
                        # + NASDAQ 100 extras
                        "NFLX", "CMCSA", "TMUS", "CHTR", "BKNG", "ADP", "MDLZ", "CSX", "MAR", "CTAS", "KLAC", "SNPS", "CDNS", "PANW", "FTNT",
                        "MELI", "NXPI", "ORLY", "ROP", "ODFL", "PCAR", "MNST", "KDP", "EXC", "XEL", "IDXX", "BIIB", "MCHP", "ALGN", "DLTR"
                    ]
                    # 중복 제거 및 정렬
                    tickers = sorted(list(set(tickers)))
                    st.caption(f"ℹ️ 속도 최적화를 위해 주요 {len(tickers)}개 우량주로 유니버스를 구성했습니다.")
                else:
                    # Full Mode: S&P 500 Constituents + NASDAQ 100 Full
                    try:
                        import FinanceDataReader as fdr
                        with st.spinner("S&P 500 및 NASDAQ 100 전종목을 병합 중입니다..."):
                            # 1. S&P 500 (Dynamic)
                            df_sp500 = fdr.StockListing('S&P500')
                            sp500_tickers = df_sp500['Symbol'].tolist()
                            
                            # 2. NASDAQ 100 (Static base + Dynamic merge)
                            # NASDAQ 100 종목들은 대부분 S&P 500에 포함되지만, 포함되지 않는 것들도 있음 (예: 일부 ADR, 비미국계 등)
                            # 두 리스트를 합칩니다.
                            
                            combined = list(set(sp500_tickers + NASDAQ_100_FULL))
                            tickers = combined
                            
                            # 500개가 넘으므로 진행 상황에 유의하라는 메시지
                            st.warning(f"⚠️ 총 {len(tickers)}개 종목 (S&P 500 + NASDAQ 100)을 분석합니다. 데이터 다운로드에 시간이 소요됩니다. (3~5분 예상)")
                    except Exception as e:
                        st.error(f"종목 리스트를 가져오는데 실패했습니다: {e}")
                        tickers = ["AAPL", "MSFT"] # Fallback

            if universe_preset != "직접 입력":
                st.info(f"선택된 유니버스: {len(tickers)}개 종목")

        with col2:
            model_type = st.selectbox(
                "사용할 AI 모델", 
                ["⭐ 앙상블 (Ensemble: Linear+SVM+LGBM)", "Linear Regression (선형회귀)", "LightGBM (트리 부스팅)", "SVM (Support Vector Machine)"]
            )
            
            # Prediction Horizon ( 보유 기간 )
            horizon_option = st.radio(
                "예측 기간 (보유 기간)",
                ["1 Day (단기 트레이딩)", "2 Weeks (스윙 트레이딩)"],
                index=1
            )
            
            # Feature 복잡도 선택
            feature_level = st.radio(
                "Feature 복잡도 (AI 지능)", 
                ["Light (5개 - 속도 중심)", "Standard (22개 - 균형)", "Rich (50+개 - 정밀 분석)", "Alpha158 (Qlib - Pro)"],
                index=1
            )
            
            # Top-K 선택
            top_k_select = st.number_input("추천할 종목 수 (Top K)", min_value=1, max_value=20, value=10)
    
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            train_start = st.date_input("학습 시작일", pd.to_datetime("2020-01-01"))
        with col_d2:
            test_start = st.date_input("테스트 시작일 (Backtest Start)", pd.to_datetime("2023-01-01"))

    # 2. 실행 (학습 버튼)
    if st.button("🧠 AI 모델 학습 시작"):
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        # A. 데이터 수집 및 피처 엔지니어링
        status_text.text("데이터 다운로드 및 피처 생성 중...")
        
        full_data = {}
        valid_tickers = []
        
        # 전체 기간 설정
        end_date = pd.to_datetime("today")
        
        for i, ticker in enumerate(tickers):
            try:
                # 넉넉하게 받아서 이평선 계산 (Rich 모드일 경우 더 많이 필요할 수 있음)
                if "Alpha158" in feature_level:
                    lookback_days = 365 # Alpha158 requires long history
                else:
                    lookback_days = 200 if "Rich" in feature_level else 100
                df = yf.download(ticker, start=train_start - pd.Timedelta(days=lookback_days), end=end_date, progress=False)
                
                # MultiIndex 처리
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # 컬럼 보정
                if 'Adj Close' not in df.columns:
                    if 'Close' in df.columns:
                        df['Adj Close'] = df['Close']
                    else:
                        continue
                
                df = df[['Open', 'High', 'Low', 'Adj Close', 'Volume']].copy()
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume'] 
                
                # [Fix] Ensure no NaNs before feature calc (Alpha158 polyfit fails on NaNs)
                df.dropna(inplace=True)

                # ---------------- [Feature Engineering (Refactored)] ----------------
                df, feature_cols = calculate_feature_set(df, feature_level)

                # Label (Target): 다음날 수익률 or 2주 후 수익률
                if "2 Weeks" in horizon_option:
                    # 10거래일 후의 수익률 (2주)
                    df['Next_Return'] = df['Close'].pct_change(10).shift(-10)
                else:
                    # 1일 후 (단기)
                    df['Next_Return'] = df['Close'].pct_change().shift(-1)
                
                df.dropna(inplace=True)
                
                if not df.empty:
                    full_data[ticker] = df
                    valid_tickers.append(ticker)
                    
            except Exception as e:
                # [Debug] Show error for first few tickers to diagnose
                if i < 3: st.warning(f"⚠️ {ticker} 처리 실패: {e}")
                pass
            
            progress_bar.progress((i + 1) / len(tickers) * 0.3)

        if not valid_tickers:
            st.error("유효한 데이터가 없습니다.")
            st.stop()
            
        # B. 모델 학습
        status_text.text(f"{model_type} 모델 학습 중 (Features: {len(feature_cols)}개)...")
        
        # 전체 데이터를 하나의 학습셋으로 병합 (Global Model)
        X_train_all = []
        y_train_all = []
        
        # feature_cols는 위에서 자동 생성됨
        
        test_datasets = {} 
        
        for ticker in valid_tickers:
            df = full_data[ticker]
            train_mask = df.index < pd.to_datetime(test_start)
            test_mask = df.index >= pd.to_datetime(test_start)
            
            train_df = df[train_mask]
            test_df = df[test_mask]
            
            if not train_df.empty:
                X_train_all.append(train_df[feature_cols].values)
                y_train_all.append(train_df['Next_Return'].values)
            
            if not test_df.empty:
                test_datasets[ticker] = test_df
        
        if not X_train_all:
            st.error("학습 데이터가 부족합니다 기간을 늘려주세요.")
            st.stop()
            
        X_train = np.concatenate(X_train_all)
        y_train = np.concatenate(y_train_all)
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Model Fitting
        if "Ensemble" in model_type:
             # 앙상블 모델 학습
            st.info("⭐ 앙상블 모드: 3가지 모델(Linear, LightGBM, SVM)을 모두 학습합니다...")
            
            # 1. Linear
            model_lin = LinearRegression()
            model_lin.fit(X_train_scaled, y_train)
            
            # 2. LightGBM
            try:
                import lightgbm as lgb
                model_lgb = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
                model_lgb.fit(X_train_scaled, y_train)
            except ImportError:
                st.warning("LightGBM not installed. Using Linear instead.")
                model_lgb = model_lin

            # 3. SVM (SVR)
            from sklearn.svm import SVR
            # 데이터가 너무 많으면 SVR은 느림. 샘플링하거나 LinearSVR 사용
            if len(X_train) > 5000:
                from sklearn.svm import LinearSVR
                model_svr = LinearSVR(random_state=42, max_iter=1000)
            else:
                model_svr = SVR(kernel='rbf')
            
            model_svr.fit(X_train_scaled, y_train)
            
            # 앙상블은 3개 모델 딕셔너리로 저장
            model = {
                "Linear": model_lin,
                "LightGBM": model_lgb,
                "SVM": model_svr
            }

        elif "Linear" in model_type:
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
        elif "SVM" in model_type:
            if len(X_train) > 10000:
                st.warning("데이터가 많아 SVM 학습 속도가 느릴 수 있습니다.")
            model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            model.fit(X_train_scaled, y_train)
        elif "LightGBM" in model_type:
            import lightgbm as lgb
            model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
            model.fit(X_train_scaled, y_train)
            
        progress_bar.progress(0.7)
        
        # C. 예측 및 백테스팅 (Dynamic Top-K)
        status_text.text(f"백테스팅 시뮬레이션 중 (Top {top_k_select})...")
        
        # 앙상블 예측 함수
        def predict_ensemble(models, X):
            p1 = models["Linear"].predict(X)
            p2 = models["LightGBM"].predict(X)
            p3 = models["SVM"].predict(X)
            # 단순 평균
            return (p1 + p2 + p3) / 3

        all_test_dates = sorted(list(set().union(*[d.index for d in test_datasets.values()])))
        
        # Holding Period Check
        holding_period = 10 if "2 Weeks" in horizon_option else 1
        rebalance_dates = all_test_dates[::holding_period]
        
        # 진행상황용
        total_steps = len(rebalance_dates) - 1
        
        cum_ret_model = 1.0
        cum_ret_bench = 1.0
        
        plot_dates = []
        plot_model = []
        plot_bench = []
        
        for i in range(total_steps):
            curr_date = rebalance_dates[i]
            next_date = rebalance_dates[i+1] # 다음 리밸런싱 날짜
            
            # 현재 시점 데이터로 예측
            candidates = []
            
            for ticker in valid_tickers:
                if ticker in test_datasets and curr_date in test_datasets[ticker].index:
                    df = test_datasets[ticker]
                    row = df.loc[curr_date]
                    
                    # Feature
                    feats = row[feature_cols].values.reshape(1, -1)
                    feats_scaled = scaler.transform(feats)
                    
                    # Score
                    if isinstance(model, dict): # Ensemble
                        score = predict_ensemble(model, feats_scaled)[0]
                    else:
                        score = model.predict(feats_scaled)[0]

                    # Actual Return (curr_date -> next_date)
                    actual_ret = 0.0
                    try:
                        p_start = df.loc[curr_date, 'Close']
                        # next_date가 없으면 그 미래 어딘가.. nearest?
                        # 단순화: next_date가 존재하면 씀. 아니면 마지막.
                        if next_date in df.index:
                            p_end = df.loc[next_date, 'Close']
                        else:
                            # next_date가 df범위를 벗어날 수도 있음 (개별 종목 상폐 등)
                            # rebalance_date logic is global, but individual ticker might end early.
                            sub_df = df.loc[curr_date:]
                            if not sub_df.empty:
                                p_end = sub_df.iloc[-1]['Close']
                            else:
                                p_end = p_start
                        
                        actual_ret = (p_end / p_start) - 1
                    except:
                        actual_ret = 0.0

                    candidates.append({
                        "ticker": ticker,
                        "score": score,
                        "ret": actual_ret
                    })
            
            if not candidates:
                continue
                
            # Score 기준 정렬
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # Top-K
            picks = candidates[:top_k_select]
            
            # 수익률 계산 (Equal Weight)
            raw_period_ret = sum([p['ret'] for p in picks]) / len(picks) if picks else 0.0
            
            # [Transaction Cost] 0.1% (0.001) per rebalance
            cost = 0.001
            period_ret = raw_period_ret - cost
            
            # Benchmark (Equal Weight of Universe)
            bench_ret = sum([p['ret'] for p in candidates]) / len(candidates) if candidates else 0.0
            
            # 누적
            cum_ret_model *= (1 + period_ret)
            cum_ret_bench *= (1 + bench_ret)
            
            plot_dates.append(next_date)
            plot_model.append(cum_ret_model)
            plot_bench.append(cum_ret_bench)
            
        progress_bar.progress(1.0)
        status_text.empty()
        
        # D. 결과 저장 (Session State)
        st.session_state.trained_models[model_type] = {
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "full_data": full_data,
            "valid_tickers": valid_tickers,
            "top_k": top_k_select,
            "feature_level": feature_level,
            "horizon": horizon_option # 저장
        }
        
        # E. 결과 시각화
        # E. 결과 시각화 & SPY Benchmark
        # SPY 데이터 가져오기
        plot_spy = [1.0] * len(plot_dates)
        try:
            spy_df = yf.download("SPY", start=plot_dates[0], end=pd.to_datetime(plot_dates[-1]) + pd.Timedelta(days=5), progress=False)
            if isinstance(spy_df.columns, pd.MultiIndex): spy_df.columns = spy_df.columns.get_level_values(0)
            target_col = 'Adj Close' if 'Adj Close' in spy_df.columns else 'Close'
            
            valid_dt = spy_df.index.asof(plot_dates[0])
            if pd.notna(valid_dt):
                base_price = spy_df.loc[valid_dt, target_col]
                temp_spy = []
                for d in plot_dates:
                    v_dt = spy_df.index.asof(d)
                    if pd.notna(v_dt):
                        temp_spy.append(spy_df.loc[v_dt, target_col] / base_price)
                    else:
                        temp_spy.append(1.0)
                plot_spy = temp_spy
        except:
            pass

        results_df = pd.DataFrame({
            "Date": plot_dates,
            "Strategy (AI)": plot_model,
            "S&P 500 (SPY)": plot_spy,
            "Benchmark (Equal)": plot_bench
        }).set_index("Date")
        
        st.success(f"학습 완료! ({model_type}) - Horizon: {horizon_option}, Top-{top_k_select}")
        
        # Prepare Backtest Data Dict for Saving
        backtest_data_to_save = {}
        
        if results_df.empty:
            st.warning("⚠️ 백테스트 기간 동안 매매 신호가 발생하지 않았거나 데이터가 부족하여 결과 그래프를 그릴 수 없습니다.")
        else:
            total_ret = results_df['Strategy (AI)'].iloc[-1] - 1
            spy_ret = results_df['S&P 500 (SPY)'].iloc[-1] - 1
            eq_ret = results_df['Benchmark (Equal)'].iloc[-1] - 1
            
            backtest_data_to_save = {
                "perf_df": results_df,
                "metrics": {
                    "Total Return": f"{total_ret:.2%}",
                    "SPY Return": f"{spy_ret:.2%}",
                    "EQ Return": f"{eq_ret:.2%}"
                }
            }

            c1, c2, c3 = st.columns(3)
            c1.metric("AI 포트폴리오 수익률 (Cost 0.1%)", f"{total_ret:.2%}")
            c2.metric("S&P 500 수익률", f"{spy_ret:.2%}")
            c3.metric("동일 비중 (Equal)", f"{eq_ret:.2%}")
            
            st.subheader(f"📈 백테스팅 결과: AI Top-{top_k_select} 전략 vs 시장")
            # st.line_chart(results_df) # Moved to Tab View

        # [Persistence Save] (Executed regardless of backtest result)
        try:
            # 앙상블은 모델 구조가 다르므로 저장 방식 유의
            model_data_to_save = {
                "model_type": model_type,
                "model": model,
                "scaler": scaler,
                "feature_cols": feature_cols,
                "feature_level": feature_level,
                "horizon": horizon_option,
                "top_k": top_k_select,
                "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                "valid_tickers": valid_tickers,
                "backtest_data": backtest_data_to_save, # Save Performance
                "train_period": f"{train_start.strftime('%Y-%m-%d')} ~ {pd.to_datetime(test_start).strftime('%Y-%m-%d')}"
            }
            
            # Update Session State too
            st.session_state.trained_models[model_type] = model_data_to_save
            
            # 파일명: {Model}_{Horizon}_{Feat}_{TopK}_{Date}.pkl
            # [Fix] Aggressive Sanitization for Windows/Cloud Compatibility
            def sanitize_filename(s):
                # Remove emojis, special chars, keep alphanumeric, spaces, hyphens, underscores
                s = re.sub(r'[^\w\s-]', '', s) # Remove non-word except space/hyphen
                return s.replace(" ", "")
            
            safe_type = sanitize_filename(model_type)
            safe_horizon = sanitize_filename(horizon_option)
            safe_feat = feature_level.split(" ")[0] # Light, Standard, Rich
            today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
            
            # [Added] Training Period in Filename (Year Only)
            period_str = f"{train_start.year}-{test_start.year}"
            
            file_name_ver = f"{safe_type}_{safe_horizon}_{safe_feat}_Top{top_k_select}_{period_str}_{today_str}"
            
            save_model_checkpoint(file_name_ver, model_data_to_save)
            st.toast(f"✅ 모델 자동 저장 완료: {file_name_ver}")
            
            # [Download Button for Git Persistence]
            saved_path = os.path.join(MODEL_SAVE_DIR, f"{file_name_ver}.pkl")
            
            st.divider() # Visual separation
            
            if os.path.exists(saved_path):
                # st.success(f"모델 저장 성공: {saved_path}") # Debug feedback
                with open(saved_path, "rb") as f:
                    btn = st.download_button(
                        label=f"📥 모델 파일 다운로드 (.pkl)\n({file_name_ver})",
                        data=f,
                        file_name=f"{file_name_ver}.pkl",
                        mime="application/octet-stream",
                        help="이 파일을 다운로드 받아 GitHub 'saved_models' 폴더에 커밋하면, Cloud 환경에서도 영구 저장됩니다.",
                        use_container_width=True # Make it prominent
                    )
            else:
                st.error(f"저장된 파일을 찾을 수 없습니다: {saved_path}")
        except Exception as e:
            st.error(f"모델 저장 실패: {e}")

    # [Fast Inference Button Logic]
    # 모델 학습 버튼 옆에 '저장된 모델 불러오기' 버튼이 있으면 좋겠지만, UI 레이아웃상
    # '2. 실행 (학습 버튼)' 아래에 조건을 두거나 병렬로 둠.
    
    # Scan for existing saved models matching current selection
    # [Fix] Use same sanitization logic as Saving to match filenames
    def sanitize_filename_search(s):
        s = re.sub(r'[^\w\s-]', '', s)
        return s.replace(" ", "")

    # [User Feedback] Scan ALL saved models, not just the selected type
    # This allows loading a 'Linear' model even if Sidebar says 'Transformer'
    safe_type = sanitize_filename_search(model_type)
    
    # Original: valid_pattern = os.path.join(MODEL_SAVE_DIR, f"*{safe_type}*.pkl")
    # New: Show all
    search_pattern = os.path.join(MODEL_SAVE_DIR, "*.pkl")
    found_files = glob.glob(search_pattern)
    
    # [Fix] Also search for 'colab_' files regardless of type selection (Universal Fallback)
    colab_pattern = os.path.join(MODEL_SAVE_DIR, "colab_*.pkl")
    colab_files = glob.glob(colab_pattern)
    
    # Union of files (avoid duplicates)
    found_files = list(set(found_files + colab_files))
    
    loaded_model_data = None
    
    if found_files:
        # Sort by modification time (newest first)
        found_files.sort(key=os.path.getmtime, reverse=True)
        
        # Pretty names for Dropdown
        # 0. Find PyTorch Models (.pth + .json)
        # Search for pairs: {name}_weights.pth and {name}_config.json
        pth_files = glob.glob(os.path.join(MODEL_SAVE_DIR, "*_weights.pth"))
        pytorch_models = {}
        for p in pth_files:
            base = p.replace("_weights.pth", "")
            config_file = f"{base}_config.json"
            if os.path.exists(config_file):
                pytorch_models[os.path.basename(base)] = {'weights': p, 'config': config_file}

        # Combine options
        # Existing .pkl files (found_files already contains correctly sorted paths from lines 1569/1575)
        file_options = {}
        
        for f in found_files:
            fname = os.path.basename(f).replace(".pkl", "")
            file_options[fname] = {'type': 'pkl', 'path': f}
            
        # Add PyTorch models to the list
        for name, paths in pytorch_models.items():
            file_options[f"[PyTorch] {name}"] = {'type': 'pth', 'path': paths['weights'], 'config': paths['config']}

        st.info(f"💡 저장된 모델: {len(file_options)}개 발견됨")
        selected_ver_key = st.selectbox("📂 불러올 모델 버전 선택", list(file_options.keys()))
        
        selected_ver = file_options.get(selected_ver_key)
        
        if selected_ver:
             try:
                 if selected_ver['type'] == 'pth':
                     # ---------------------------------------------------------
                     # [Safe] Load Pure PyTorch Model (No Qlib Required)
                     # ---------------------------------------------------------
                     import torch
                     import torch.nn as nn
                     import json
                     import math

                     st.toast("🛡️ 안전 모드: 순수 PyTorch 모델을 로딩합니다.")

                     # 1. Load Config
                     with open(selected_ver['config'], 'r') as f:
                         conf = json.load(f)

                     # 2. Define Pure PyTorch Transformer Class (Indentical to Qlib's logic)
                     class PositionalEncoding(nn.Module):
                        def __init__(self, d_model, max_len=1000):
                            super(PositionalEncoding, self).__init__()
                            pe = torch.zeros(max_len, d_model)
                            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                            pe[:, 0::2] = torch.sin(position * div_term)
                            pe[:, 1::2] = torch.cos(position * div_term)
                            # [Fix] Shape match: [1000, 1, 64]
                            self.register_buffer('pe', pe.unsqueeze(1))
                        def forward(self, x):
                            return x + self.pe[:x.size(0), :]

                     class TransformerModel(nn.Module):
                        def __init__(self, d_feat, d_model=64, nhead=8, num_layers=2, dropout=0.1, device='cpu'):
                            super(TransformerModel, self).__init__()
                            self.feature_layer = nn.Linear(d_feat, d_model)
                            self.pos_encoder = PositionalEncoding(d_model)
                            # [Fix] dim_feedforward=2048 (Default)
                            encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, 2048, dropout)
                            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
                            self.decoder_layer = nn.Linear(d_model, 1)
                            self.device = device
                            self._d_feat_val = d_feat
                        def forward(self, src):
                            src = self.feature_layer(src)
                            src = self.pos_encoder(src)
                            output = self.transformer_encoder(src)
                            output = self.decoder_layer(output[:, -1, :]) # Use last step
                            return output.squeeze()
                        # Add d_feat property for auto-detection
                        @property
                        def d_feat(self):
                            return self._d_feat_val
                        def __init__(self, d_feat, d_model=64, nhead=8, num_layers=2, dropout=0.1, device='cpu'):
                            super(TransformerModel, self).__init__()
                            self._d_feat_val = d_feat # Store for property
                        def predict(self, dataset):
                            # Adapter for DataFrame or Numpy
                            val = None
                            if isinstance(dataset, pd.DataFrame):
                                val = dataset.values
                            elif isinstance(dataset, np.ndarray):
                                val = dataset
                            
                            if val is not None:
                                x = torch.tensor(val, dtype=torch.float32).to(self.device)
                                if len(x.shape) == 2:
                                    x = x.unsqueeze(1) # [Batch, 1, Feat]
                                self.eval()
                                with torch.no_grad():
                                    pred = self.forward(x)
                                return pred.cpu().numpy()
                            return []

                     # 3. Initialize Model
                     model = TransformerModel(
                         d_feat=conf.get('d_feat', 20), 
                         d_model=conf.get('d_model', 64),
                         nhead=conf.get('nhead', 8),
                         num_layers=conf.get('num_layers', 2),
                         device='cpu' 
                     )
                     
                     # 4. Load Weights
                     # [Fix] strict=False to ignore 'encoder_layer.*' keys which are unused prototype layers in Qlib
                     model.load_state_dict(torch.load(selected_ver['path'], map_location=torch.device('cpu')), strict=False)
                     model.eval()
                     
                     # 5. Wrap as standard object
                     loaded_model_data = {
                         "model": model,
                         "feature_level": "Standard", # Default
                         "description": "PyTorch Transformer (Imported)",
                         "scaler": None, # [Fix] Add default scaler
                         "feature_cols": [] # [Fix] Add default empty list, will be populated on fly
                     }
                     
                 else:
                     # Traditional Pickle Loading (Mock Logic)
                     with open(selected_ver['path'], "rb") as f:
                         loaded_model_data = pickle.load(f)

             except Exception as e:
                 st.error(f"모델 로드 실패: {e}") 
                 if 'qlib' in str(e):
                      st.error("💡 Qlib 관련 에러입니다. 순수 PyTorch 모드(.pth)를 사용해보세요.")

    if loaded_model_data:
        saved_ts = loaded_model_data.get('timestamp', 'Unknown')
        
        # [UX Improvement] Pre-Inference Model Specs
        st.info("ℹ️ 선택된 모델 상세 정보 (Model Specs)")
        spec_col1, spec_col2 = st.columns(2)
        with spec_col1:
            st.write(f"**🔹 모델 타입**: {loaded_model_data.get('model_type')}")
            st.write(f"**🔹 예측 기간 (Horizon)**: {loaded_model_data.get('horizon')}")
            st.write(f"**🔹 학습 기간**: {loaded_model_data.get('train_period', 'Unknown')}")
        
        with spec_col2:
            feat_lvl = loaded_model_data.get('feature_level', 'Unknown')
            feat_cnt = len(loaded_model_data.get('feature_cols', []))
            st.write(f"**🔹 Feature 복잡도**: {feat_lvl} ({feat_cnt} features)")
            univ_size = len(loaded_model_data.get('valid_tickers', []))
            st.write(f"**🔹 학습 유니버스 크기**: {univ_size}개 종목")
            st.write(f"**🔹 저장 일시**: {saved_ts}")
            
        st.divider()

        # [UX Improvement] Add Top-K slider specific for Inference here
        st.write("#### ⚙️ 추론 설정 (Inference Settings)")
        top_k_inference = st.slider("추천할 종목 수 (Top K)", min_value=1, max_value=50, value=10, key="top_k_inf")
        
        # [Fix] Robust feature level retrieval
        feat_level = loaded_model_data.get('feature_level', 'Standard')
        
        # Action Buttons (Inference & Delete)
        col_act1, col_act2 = st.columns([3, 1])
        
        with col_act2:
             if st.button("🗑️ 모델 삭제 (Delete)", type="primary"):
                 try:
                     os.remove(selected_ver['path'])
                     st.toast("✅ 모델 파일이 삭제되었습니다.")
                     st.rerun()
                 except Exception as e:
                     st.error(f"삭제 실패: {e}")

        with col_act1:
            run_inference = st.button("⚡ 선택된 모델로 바로 분석 (Fast Inference)", type="primary")

        if run_inference:
            # Inject Backtest Data for Analysis Tab
            if 'backtest_data' in loaded_model_data:
                 st.session_state.trained_models[model_type] = loaded_model_data
                 
                 # [Fix] Render Saved Backtest Results Immediately
                 bd = loaded_model_data['backtest_data']
                 if bd and 'perf_df' in bd and not bd['perf_df'].empty:
                     metrics = bd.get('metrics', {})
                     res_df = bd['perf_df']
                     
                     st.markdown("### 📊 불러온 모델의 백테스트 결과")
                     c1, c2, c3 = st.columns(3)
                     c1.metric("AI 포트폴리오 수익률", metrics.get("Total Return", "N/A"))
                     c2.metric("S&P 500 수익률", metrics.get("SPY Return", "N/A"))
                     c3.metric("동일 비중 (Equal)", metrics.get("EQ Return", "N/A"))
                     
                     st.line_chart(res_df)
                 else:
                     st.warning("⚠️ 저장된 백테스트 데이터가 없습니다.")
                 
                 # [Supported Request] Display Model Characteristics
                 st.info("ℹ️ 모델 상세 스펙 (Model Specifications)")
                 
                 spec_col1, spec_col2 = st.columns(2)
                 with spec_col1:
                     st.write(f"**🔹 모델 타입**: {loaded_model_data.get('model_type')}")
                     st.write(f"**🔹 예측 기간 (Horizon)**: {loaded_model_data.get('horizon')}")
                     st.write(f"**🔹 학습 기간**: {loaded_model_data.get('train_period', 'Unknown (Old Version)')}")
                 
                 with spec_col2:
                     feat_lvl = loaded_model_data.get('feature_level', 'Unknown')
                     feat_cnt = len(loaded_model_data.get('feature_cols', []))
                     st.write(f"**🔹 Feature 복잡도**: {feat_lvl} ({feat_cnt} features)")
                     
                     univ_size = len(loaded_model_data.get('valid_tickers', []))
                     st.write(f"**🔹 학습 유니버스 크기**: {univ_size}개 종목")
                     st.write(f"**🔹 저장 일시**: {saved_ts}")
                 st.divider()

            status_text = st.empty()
            progress_bar = st.progress(0)
            status_text.text("최신 데이터 다운로드 중 (Fast Mode - Last 200 Days)...")
            
            # Load Params
            model = loaded_model_data['model']
            scaler = loaded_model_data['scaler']
            feature_cols = loaded_model_data['feature_cols']
            saved_tickers = loaded_model_data.get('valid_tickers', [])
            
            # Use current tickers logic? Or saved? 
            # User wants to run on CURRENT universe but with SAVED model?
            # Generally, model trained on Tickers A,B,C might not work well on D,E,F if features are generic enough?
            # AI models are trained on patterns. Features (MA, RSI) are generic.
            # So we can apply saved model to NEW universe or CURRENT universe selection.
            # Let's use the CURRENT UI selection 'tickers' to be flexible.
            target_tickers = tickers if tickers else saved_tickers
            
            fast_data = {}
            fast_valid_tickers = []
            
            # Fast Download (Short period)
            end_date = pd.to_datetime("today")
            # Feature calculation needs ~120 days buffer
            start_date_fast = end_date - pd.Timedelta(days=365) # 1 year safe buffer
            
            for i, ticker in enumerate(target_tickers):
                try:
                    df = yf.download(ticker, start=start_date_fast, end=end_date, progress=False)
                    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                    if 'Adj Close' not in df.columns:
                        if 'Close' in df.columns: df['Adj Close'] = df['Close']
                        else: continue
                    
                    df = df[['Open', 'High', 'Low', 'Adj Close', 'Volume']].copy()
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    
                    # [Auto-Detect] Alpha158 Requirement
                    target_level = loaded_model_data.get('feature_level', 'Standard')
                    expected_dim = getattr(model, 'd_feat', None)
                    if expected_dim == 158:
                         target_level = "Alpha158"
                         if i==0: st.toast("🧪 Alpha158 Features Auto-Detected & Applied", icon="⚡")
                    
                    # Feature Engineer
                    df, _ = calculate_feature_set(df, target_level)
                    
                    # [Fix] Auto-detect feature columns if missing (for PyTorch models)
                    if not feature_cols:
                         # For Alpha158, the function returns cols.
                         # If calculate_feature_set returned cols, use them.
                         # But wait, calculate_feature_set returns (df, cols).
                         # We captured it in _. Wait, code above says: df, _ = ...
                         # We MUST capture feature_cols!
                         pass # handled below
                         
                    # Re-retrieve feature_cols from calculation if needed
                    # Note: calculate_feature_set returns (df, feature_cols)
                    # We should use that value.
                    
                    # Let's fix the call signature interaction
                    df, computed_cols = calculate_feature_set(df, target_level)
                    if not feature_cols:
                        feature_cols = computed_cols
                        loaded_model_data['feature_cols'] = feature_cols
                    else:
                        # Validation: if dimension differs, prefer computed
                        if len(feature_cols) != len(computed_cols) and len(computed_cols) == 158:
                             feature_cols = computed_cols
                    
                    
                    # Drop NaN
                    df.dropna(inplace=True)
                    
                    if not df.empty:
                        fast_data[ticker] = df
                        fast_valid_tickers.append(ticker)
                        
                except Exception as e:
                    # [Debug] Show error if prediction fails
                    if i < 3: # Show only first few errors to avoid spam
                         st.error(f"Error for {ticker}: {e}")
                    pass
                progress_bar.progress((i+1)/len(target_tickers))
            
            status_text.text("AI 모델 예측 수행 중...")
            
            # Prepare Session State for Results (Mocking the 'trained_models' state for the result viewer)
            # But wait, result viewer expects full_data, etc. 
            # We should populate session_state exactly as if we trained.
            
            st.session_state.trained_models[model_type] = {
                "model": model,
                "scaler": scaler,
                "feature_cols": feature_cols,
                "full_data": fast_data, # Only recent data
                "valid_tickers": fast_valid_tickers,
                "top_k": top_k_inference, # [Fix] Use the inference specific slider
                "feature_level": loaded_model_data['feature_level'],
                "horizon": horizon_option
            }
            
            st.success(f"⚡ 빠른 분석 완료! 하단 '오늘의 추천 PICK'에서 결과를 확인하세요.")
            status_text.empty()
            
            # -----------------------------------------------------------------------------
            # [Fix] Result Rendering for Fast Inference (Same logic as Training)
            # -----------------------------------------------------------------------------
            st.divider()
            st.subheader("🚀 오늘의 추천 PICK (Latest Predictions)")
            
            # A. Generate Predictions for the Latest Date
            recommendations = []
            
            st.write(f"🔎 Valid Tickers for Inference: {len(fast_valid_tickers)}") # Debug
            
            for ticker in fast_valid_tickers:
                df = fast_data[ticker]
                if df.empty: continue
                
                # Get Last Row
                last_row = df.iloc[[-1]] # Keep DataFrame format
                
                # Prepare Features
                try:
                    feat_vals = last_row[feature_cols].copy() # Ensure DataFrame/Series
                    
                    # 1. Scale (if scaler exists)
                    if scaler:
                         feat_scaled = scaler.transform(feat_vals.values)
                    else:
                         feat_scaled = feat_vals.values
                    
                    # 2. [Critical] Check Feature Dimension (Padding for Alpha158 vs Standard mismatch)
                    expected_dim = getattr(model, 'd_feat', None)
                    if expected_dim and feat_scaled.shape[1] < expected_dim:
                         current_dim = feat_scaled.shape[1]
                         pad_size = expected_dim - current_dim
                         import numpy as np
                         padding = np.zeros((feat_scaled.shape[0], pad_size))
                         feat_scaled = np.hstack([feat_scaled, padding])
                    
                    # Debug: Show shape and first few values
                    # if ticker == "AAPL":
                    #      st.write(f"AAPL Shape: {feat_scaled.shape}")
                    #      st.write(f"AAPL Feats: {feat_scaled[0][:5]}...")
                    
                    # Predict
                    score = 0
                    if isinstance(model, dict): # Ensemble
                         if "Linear" in model: 
                             p = model["Linear"].predict(feat_scaled)
                             score += p.item() if hasattr(p, 'item') else p[0]
                         if "LightGBM" in model: 
                             p = model["LightGBM"].predict(feat_scaled)
                             score += p.item() if hasattr(p, 'item') else p[0]
                         if "SVM" in model: 
                             p = model["SVM"].predict(feat_scaled)
                             score += p.item() if hasattr(p, 'item') else p[0]
                         score /= 3.0
                    else:
                        pred_res = model.predict(feat_scaled)
                        # Robust scalar extraction
                        if hasattr(pred_res, 'item'):
                            score = pred_res.item()
                        elif hasattr(pred_res, '__iter__') and len(pred_res) > 0:
                            score = pred_res.flat[0] # Handle any shape
                        else:
                            score = pred_res # Assume simple float
                    
                    # Debug Score
                    st.write(f"{ticker} Score: {score} (Type: {type(score)})")
                        
                    # Interpret Score
                    signal = "Hold (관망)"
                    if score > 0.01: signal = "Strong Buy (강력 매수) 🚀"
                    elif score > 0.005: signal = "Buy (매수) 📈"
                    elif score < -0.01: signal = "Strong Sell (강력 매도) 📉"
                    elif score < -0.005: signal = "Sell (매도) 🔻"
                    
                    recommendations.append({
                        "종목코드": ticker,
                        "🚦 매매 신호": signal,
                        "📈 예상 등락률": f"{score:.2%}", # Format as %
                        "Raw_Score": score, # Hidden for sorting
                        "현재가": f"{last_row['Close'].values[0]:,.0f}",
                        "기준일": last_row.index[-1].strftime('%Y-%m-%d')
                    })
                except Exception as e:
                    # st.error(f"Inference Logic Error for {ticker}: {e}")
                    pass
            
            # Save Results to Session State for Display
            if recommendations:
                st.session_state.trained_models[model_type]['cached_recommendations'] = recommendations
                st.toast(f"✅ Analysis Complete! Found {len(recommendations)} stocks.", icon="🎉")
            else:
                 st.warning("No recommendations generated.")

            pass

    elif 'scan_results' in st.session_state and not st.session_state.scan_results:
         st.info("현재 기준 특이 패턴(골든크로스, 과매수/과매도 등)이 발견된 종목이 없습니다.")


    # -----------------------------------------------------------------------------
    # Unified Result Rendering (Tabs) - Runs for both Training & Inference
    # -----------------------------------------------------------------------------
    if model_type in st.session_state.trained_models:
        st.divider()
        st.subheader("📊 AI 모델 분석 결과")
        
        # Load Data
        model_info = st.session_state.trained_models[model_type]
        
        tab1, tab2, tab3 = st.tabs(["📈 성과 분석 (Analysis)", "📋 오늘의 추천 (Recommendations)", "📜 히스토리 (History)"])
        
        # TAB 1: Analysis (Backtest)
        with tab1:
            st.markdown("### 📈 백테스팅 성과 (Backtest Performance)")
            
            # Check for Backtest Data
            bd = model_info.get('backtest_data', {})
            if bd and 'perf_df' in bd and not bd['perf_df'].empty:
                 metrics = bd.get('metrics', {})
                 res_df = bd['perf_df']
                 
                 c1, c2, c3 = st.columns(3)
                 c1.metric("AI 포트폴리오 수익률", metrics.get("Total Return", "N/A"))
                 c2.metric("S&P 500 수익률", metrics.get("SPY Return", "N/A"))
                 c3.metric("동일 비중 (Equal)", metrics.get("EQ Return", "N/A"))
                 
                 st.line_chart(res_df)
            else:
                 st.info("백테스트 데이터가 없습니다 (Short-term Inference Mode 등).")

        # TAB 2: Recommendations (Top-K)
        with tab2:
            st.markdown("### 📋 오늘의 Top Picks")
            
            # Priority: Use Cached Recommendations from Fast Inference
            if 'cached_recommendations' in model_info and model_info['cached_recommendations']:
                 recs = model_info['cached_recommendations']
                 # Convert to DataFrame
                 rec_df = pd.DataFrame(recs)
            else:
                 # Fallback: Training-time Inference (Original Logic)
                 current_model = model_info['model']
                 # ... (Old Loop Logic would go here if we wanted fallback, but for now we rely on cache)
                 rec_df = pd.DataFrame() # Empty
            
            if not rec_df.empty:
                 current_top_k = model_info.get('top_k', 5)
                 # Sort and Limit (Use Raw_Score for sorting)
                 final_picks = rec_df.sort_values(by="Raw_Score", ascending=False).head(current_top_k).copy()
                 final_picks = final_picks.reset_index(drop=True)
                 final_picks.index += 1 # 1-based index
                 
                 st.markdown(f"### 🚀 오늘의 Top-{len(final_picks)} 추천 종목")
                 
                 # Display Friendly DataFrame
                 display_cols = ["종목코드", "🚦 매매 신호", "📈 예상 등락률", "현재가", "기준일"]
                 st.dataframe(final_picks[display_cols], use_container_width=True)
                 
                 st.caption("ℹ️ **예상 등락률**: AI 모델이 예측한 다음 보유 기간(Horizon) 동안의 수익률입니다.")
                 st.caption("ℹ️ **매매 신호**: 예상 등락률이 1% 이상이면 '강력 매수', 0.5% 이상이면 '매수'로 분류합니다.")
                 
                 # Save Portfolio Button
                 if st.button("💾 이 포트폴리오 저장하기 (Save History)"):
                     from datetime import datetime
                     save_entry = {
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model_type": model_type,
                        "holdings": final_picks.to_dict(orient='records'),
                        "top_k": len(final_picks),
                        "feature_level": model_info.get('feature_level', 'Unknown'),
                        "horizon": model_info.get('horizon', 'Unknown')
                     }
                     # Load existing & Append
                     hist = load_portfolio_history()
                     if not isinstance(hist, list): hist = []
                     hist.append(save_entry)
                     save_portfolio_history(hist)
                     st.success("포트폴리오가 이력에 저장되었습니다.")
            else:
                 st.info("아직 추천 결과가 생성되지 않았습니다. 사이드바에서 'Fast Inference' 버튼을 눌러주세요.")
                
        # TAB 3: History
        with tab3:
            st.markdown("### 📜 나의 포트폴리오 저장 이력")
            
            # [Restored Feature] Import History
            with st.expander("📂 백업 파일에서 불러오기 (Import History)", expanded=False):
                st.caption("기존에 다운로드 받았던 `portfolio_history.json` 파일을 업로드하면 복원됩니다.")
                uploaded_file = st.file_uploader("JSON 파일 선택", type=["json"], key="history_uploader")
                
                if uploaded_file is not None:
                    try:
                        imported_data = json.load(uploaded_file)
                        if isinstance(imported_data, list):
                            # Append to current history
                            current_hist = load_portfolio_history()
                            if not isinstance(current_hist, list): current_hist = []
                            
                            # Deduplication check (simple date check)
                            existing_dates = {x.get('date') for x in current_hist}
                            count = 0
                            for item in imported_data:
                                if item.get('date') not in existing_dates:
                                    current_hist.append(item)
                                    count += 1
                                    
                            if count > 0:
                                save_portfolio_history(current_hist)
                                st.success(f"✅ {count}개의 포트폴리오 이력이 복원되었습니다. 화면을 새로고침합니다.")
                                st.rerun()
                            else:
                                st.warning("이미 존재하는 이력들입니다.")
                        else:
                            st.error("올바르지 않은 JSON 형식입니다.")
                    except Exception as e:
                        st.error(f"파일 불러오기 실패: {e}")
            
            # Load History
            hist_data = load_portfolio_history()
            
            if not hist_data:
                st.info("아직 저장된 포트폴리오가 없습니다.")
            else:
                # History Download Button
                json_str = json.dumps(hist_data, default=str, indent=4)
                st.download_button(
                    label="💾 전체 이력 다운로드 (Backup)",
                    data=json_str,
                    file_name=f"portfolio_history_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
                st.divider()

                # Show list
                # Reverse order
                if isinstance(hist_data, list):
                    for idx, item in enumerate(reversed(hist_data)):
                        idx_real = len(hist_data) - 1 - idx
                        with st.expander(f"📅 {item.get('date', 'Unknown')} - {item.get('model_type')} ({len(item.get('holdings', []))} Stocks)"):
                            c_del, c_view = st.columns([1, 5])
                            with c_del:
                                if st.button("🗑️ 삭제", key=f"del_hist_{idx_real}"):
                                    del hist_data[idx_real]
                                    save_portfolio_history(hist_data)
                                    st.rerun()
                                    
                            st.write(f"**Top-K**: {item.get('top_k')} | **Horizon**: {item.get('horizon')}")
                            h_df = pd.DataFrame(item.get('holdings', []))
                            st.dataframe(h_df)
                else:
                    st.error("데이터 형식이 올바르지 않습니다.")
                    



# -----------------------------------------------------------------------------
# 🔎 ETF 구성 종목 검색 (Reverse Search)
# -----------------------------------------------------------------------------
elif selection == "🔎 ETF 구성 종목 검색":
    st.title("🔎 ETF 구성 종목 검색 (Reverse Search)")
    st.caption("특정 종목을 담고 있는 ETF를 검색하고, 비중 순으로 정렬합니다. (KRX 실시간 데이터 기반)")

    import FinanceDataReader as fdr

    # 1. 유효한 데이터가 있는 최신 영업일 구하기
    # (주의: fdr은 별도 날짜 체크 없이 최신 리스트를 가져오므로, 여기서는 단순히 오늘 날짜 또는 안전한 평일을 반환)
    @st.cache_data(ttl=3600*12) 
    def get_latest_biz_date():
        # ETF PDF 데이터를 가져올 때는 날짜가 중요하므로, 평일인지 체크
        curr = datetime.now()
        # 만약 주말이면 금요일로 이동
        while curr.weekday() > 4:
            curr -= timedelta(days=1)
        return curr.strftime("%Y%m%d")

    target_date = get_latest_biz_date()
    st.info(f"📅 데이터 기준일: **{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}** (KRX)")

    # 2. 데이터 수집 및 캐싱
    @st.cache_data(ttl=3600*24, show_spinner=False) # 24시간 캐시
    def get_all_etf_data(date):
        """
        모든 ETF의 구성 종목(PDF) 데이터를 수집하여 Dictionary 형태로 반환합니다.
        Key: Ticker, Value: Data (Name, PDF_DataFrame)
        """
        # A. ETF 리스트 가져오기 (pykrx 대신 fdr 사용 - 인코딩 이슈 우회)
        tickers = []
        try:
            # KRX ETF 리스트 (Symbol, Name 등 포함)
            etf_list_df = fdr.StockListing('ETF/KR')
            tickers = etf_list_df['Symbol'].tolist()
            
            # [테스트 모드] 너무 오래 걸리므로 상위 20개만 우선 테스트
            # 테스트 완료 후 아래 두 줄 주석 처리 또는 삭제하면 전체 다운로드 가능
            if len(tickers) > 20: 
                 tickers = tickers[:20] 
                 st.info(f"⚡ [테스트 모드] 빠른 확인을 위해 전체 {len(etf_list_df)}개 중 **상위 20개 ETF**만 스캔합니다.")
        except Exception as e:
            # st.error(f"ETF 리스트를 가져오는 중 오류 발생 (FDR): {e}")
            pass
        
        # Fallback: 리스트 가져오기 실패 시 주요 ETF 하드코딩
        if not tickers:
            tickers = [
                "069500", # KODEX 200
                "371460", # TIGER 차이나전기차SOLACTIVE
                "122630", # KODEX 레버리지
                "252670", # KODEX 200선물인버스2X
                "233740", # KODEX 코스닥150레버리지
                "251340", # KODEX 코스닥150선물인버스
                "102110", # TIGER 200
                "278530", # KODEX 200TR
                "278540", # TIGER 200TR
                "360750", # TIGER 미국S&P500
                "360200", # TIGER 미국나스닥100
            ]
            st.warning("⚠️ ETF 전체 리스트를 가져오지 못해 주요 11개 ETF만 스캔합니다.")
            # st.success(f"총 {len(tickers)}개의 ETF 리스트를 확보했습니다.")
            pass
            
        etf_data = {}
        error_count = 0


        
        # 진행률 표시 (최초 실행 시에만 보임)
        progress_text = "KRX에서 모든 ETF 데이터(PDF)를 수집 중입니다... (최초 1회 실행 시 3~5분 소요)"
        my_bar = st.progress(0, text=progress_text)
        
        total = len(tickers)
        
        # FDR에서 가져온 이름 매핑 (Name Column 확인 필요, 보통 'Name')
        name_map = {}
        if 'Name' in etf_list_df.columns:
            name_map = etf_list_df.set_index('Symbol')['Name'].to_dict()
        
        total = len(tickers)
        
        
        last_error = None
        
        for i, ticker in enumerate(tickers):
            pdf = None
            try:
                # 1. pykrx 시도
                try:
                    pdf = stock.get_etf_portfolio_deposit_file(ticker, date)
                except:
                    pdf = None

                # 2. 실패 시: Daum Finance API (Kakao Pay) - 차단 가능성 낮음
                if pdf is None or pdf.empty:
                    try:
                        # Daum URL: https://finance.daum.net/api/etf/constituents?symbolCode=A069500
                        # Ticker에 'A' 붙여야 함
                        daum_ticker = f"A{ticker}"
                        url = f"https://finance.daum.net/api/etf/constituents?symbolCode={daum_ticker}"
                        
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                            'Referer': f'https://finance.daum.net/quotes/{daum_ticker}',
                            'Accept': 'application/json, text/plain, */*',
                            'Host': 'finance.daum.net'
                        }
                        
                        resp = requests.get(url, headers=headers, verify=False, timeout=5)
                        
                        if resp.status_code == 200:
                            data_json = resp.json()
                            if "data" in data_json:
                                holdings = data_json["data"]
                                temp_df = pd.DataFrame(holdings)
                                
                                if not temp_df.empty:
                                    # 컬럼 매핑 needed
                                    # Daum fields: symbolCode, name, tradePrice, weight
                                    rename_map = {
                                        'name': 'Name',
                                        'weight': '비중',
                                        'tradePrice': '금액', # 정확히는 현재가지만 금액 대용으로 사용 가능 여부 확인. 
                                        # 하지만 ETF PDF의 '금액'은 '보유금액'이므로 tradePrice(현재가)와 다름.
                                        # 비중이 핵심.
                                        'symbolCode': 'Code'
                                    }
                                    pdf = temp_df.rename(columns=rename_map)
                                    # Code에서 'A' 제거 (A005930 -> 005930)
                                    if 'Code' in pdf.columns:
                                        pdf['Code'] = pdf['Code'].str.replace('A', '', regex=False)
                                    
                                    # 비중이 0~100 사이 숫자인지 확인. Daum은 보통 0.25 (1% 미만) or 25.0 ?
                                    # 확인 결과 Daum은 0.15 (=0.15%) 식으로 줄 수도 있고 15.0일 수도 있음. 
                                    # 일단 그대로 둠.
                                    
                                    if '금액' not in pdf.columns:
                                         pdf['금액'] = 0
                            
                    except Exception as e_daum:
                        if last_error is None:
                            last_error = f"Daum API Error: {str(e_daum)}"
                        pass

                # 3. 최후의 수단: Yahoo Finance (yfinance) - 해외 IP(Streamlit Cloud)에서 작동 가능
                if pdf is None or pdf.empty:
                    try:
                        # Ticker format: 069500.KS
                        yf_ticker = f"{ticker}.KS"
                        fund = yf.Ticker(yf_ticker)
                        
                        # Top Holdings 가져오기 (보통 상위 10개만 제공됨)
                        holdings_df = None
                        try:
                            # funds_data.top_holdings는 pandas DF 리턴 (Name, Symbol, Holding Percent 등)
                            # yfinance 버전에 따라 다를 수 있음. 최신버전 기준 시도.
                            # 혹은 info['holdings'] 등
                            # 여기선 안전하게 funds_data 접근 시도
                            if hasattr(fund, 'funds_data') and fund.funds_data:
                                holdings_df = fund.funds_data.top_holdings
                        except:
                            pass
                            
                        if holdings_df is not None and not holdings_df.empty:
                            # 컬럼 매핑: Name, Symbol, Holding % (0.05 form or 5.0 form)
                            # yfinance returns: index=Symbol, columns=['Name', 'Holding %', 'Buying', 'Selling']
                            # Reset index to get Symbol as column
                            holdings_df = holdings_df.reset_index()
                            
                            rename_map = {
                                'Name': 'Name',
                                'Symbol': 'Code',
                                'Holding %': '비중' 
                            }
                            # 컬럼이 다를 수 있으니 확인
                            cols = holdings_df.columns
                            if 'Name' in cols and 'Holding %' in cols:
                                pdf = holdings_df.rename(columns=rename_map)
                                # 비중이 0.xx 형태면 * 100 해야함 (yfinance는 보통 0.0524 형태로 줌)
                                # 근데 yf 최신은 이미 %단위(5.24)일 수도 있음. 확인 필요. 
                                # 보통 funds_data는 0~1 scale인 경우가 많음 -> 확인 불가하므로 그대로 둠
                                # 금액 정보 없음
                                pdf['금액'] = 0
                                
                    except Exception as e_yf:
                        if last_error is None:
                             last_error = f"Yahoo Error: {str(e_yf)}"
                        pass
                
                # 데이터 유효성 검사
                if pdf is not None and not pdf.empty:
                    # FDR Name Map 사용
                    name = name_map.get(str(ticker), str(ticker))
                    
                    etf_data[ticker] = {
                        "name": name,
                        "pdf": pdf 
                    }
            except Exception as e:
                error_count += 1
            
            # 진행률 업데이트 (너무 자주하면 느려지므로 5% 단위 or 10개 단위)
            if i % 10 == 0:
                my_bar.progress((i + 1) / total, text=f"{progress_text} ({i+1}/{total})")
                
        my_bar.empty()
        
        # Debug Info: 첫번째 에러 보여주기 (사용자 피드백용)
        if last_error and not etf_data:
             with st.expander("⚠️ 데이터 수집 에러 상세 (Debug Logs)"):
                st.write(f"Last Error: {last_error}")

        
        if error_count > 0:
            st.warning(f"{error_count}개의 ETF 데이터를 가져오는 데 실패했습니다 (상장폐지 등 이유).")
            
        return etf_data

    # 데이터 로딩 Trigger
    with st.spinner("데이터베이스를 동기화 중입니다... 잠시만 기다려주세요."):
        all_etf_data = get_all_etf_data(target_date)

    # 3. 검색 UI
    st.divider()
    search_query = st.text_input("검색할 종목명을 입력하세요 (예: 삼성전자, NAVER)", placeholder="종목명 입력 후 Enter").strip()

    if search_query:
        # A. 검색 로직
        found_etfs = []
        
        # 사용자가 입력한 게 티커인지 이름인지 모름 -> 이름으로 매칭 시도
        # pykrx의 PDF 데이터에는 종목코드가 인덱스이고, 종목명은 없을 수 있음.
        # 따라서 "삼성전자"를 "005930"으로 변환하거나, PDF 내에 종목명이 있는지 확인해야 함.
        # get_etf_portfolio_deposit_file() 결과는 보통 인덱스=티커, 컬럼=[계약수, 금액, 비중] 형태임. 종목명이 없음.
        # 해결책:
        # 1. KOSPI/KOSDAQ 전 종목 마스터 데이터를 가져와서 {이름: 티커} 매핑을 만듦.
        # 2. 사용자가 입력한 "삼성전자" -> "005930" 변환.
        # 3. 각 ETF의 PDF 인덱스(티커)에 "005930"이 있는지 확인.
        
        @st.cache_data
        def get_stock_name_map(date):
            # 1. FDR KRX 전체 리스트 시도
            name_map = {}
            try:
                df_krx = fdr.StockListing('KRX')
                if not df_krx.empty:
                    name_map = df_krx.set_index('Name')['Symbol'].to_dict()
            except Exception as e:
                pass
            
            # 2. 실패하거나 비어있으면 KOSPI/KOSDAQ 개별 시도 (Fallback)
            if not name_map:
                try:
                    df_kospi = fdr.StockListing('KOSPI')
                    df_kosdaq = fdr.StockListing('KOSDAQ')
                    if not df_kospi.empty:
                        name_map.update(df_kospi.set_index('Name')['Symbol'].to_dict())
                    if not df_kosdaq.empty:
                        name_map.update(df_kosdaq.set_index('Name')['Symbol'].to_dict())
                except:
                    pass
            # 3. 최후의 수단: 주요 종목 하드코딩 (네트워크/파싱 전면 실패 시 대비)
            if not name_map:
                name_map = {
                    "삼성전자": "005930",
                    "SK하이닉스": "000660",
                    "NAVER": "035420",
                    "카카오": "035720",
                    "LG에너지솔루션": "373220",
                    "현대차": "005380",
                    "POSCO홀딩스": "005490",
                    "기아": "000270",
                    "KB금융": "105560"
                }
            
            return name_map

        name_map = get_stock_name_map(target_date)
        
        # Debug Info: 활성화해서 상태 확인
        st.warning(f"🔍 Debug Info: Loaded {len(name_map)} stocks. " 
                   f"Sample: {list(name_map.keys())[:5] if name_map else 'Empty'}")

        
        # 검색어 매칭 (정확치 & 포함)
        target_ticker = name_map.get(search_query) # 정확히 일치
        
        # 정확히 일치하지 않으면 포함 검색 (첫 번째 발견된 것)
        if not target_ticker:
            candidates = [name for name in name_map.keys() if search_query.upper() in name.upper()]
            if len(candidates) > 0:
                # 선택지 제공? 아니면 첫번째?
                # UX상 모호하면 가장 유사한 것 선택 or Selectbox
                if len(candidates) == 1:
                    target_ticker = name_map[candidates[0]]
                    st.success(f"'{candidates[0]}' ({target_ticker}) 종목으로 검색합니다.")
                else:
                    st.info(f"검색어 '{search_query}'와 유사한 종목: {', '.join(candidates[:5])} ...")
                    selected_name = st.selectbox("종목을 선택하세요:", candidates)
                    target_ticker = name_map[selected_name]
            else:
                st.error("해당하는 종목을 찾을 수 없습니다.")
                st.stop()
        
        # B. ETF 필터링
        result_list = []
        
        # Debug: 데이터가 비어있는지 확인
        if not all_etf_data:
            st.error("ETF 데이터를 하나도 수집하지 못했습니다. (KRX/네이버 접속 실패)")
        else:
            # st.info(f"Debug: {len(all_etf_data)}개 ETF 데이터 스캔 중...")
            pass

        for etf_ticker, data in all_etf_data.items():
            pdf_df = data['pdf']
            found = False
            row = None
            
            # 1. Ticker로 검색 (pykrx 데이터인 경우 Index가 Ticker)
            if target_ticker in pdf_df.index:
                row = pdf_df.loc[target_ticker]
                found = True
            
            # 2. Ticker가 컬럼에 있는지 확인
            elif 'Code' in pdf_df.columns and target_ticker in pdf_df['Code'].values:
                # 해당 로우 찾기
                row = pdf_df[pdf_df['Code'] == target_ticker].iloc[0]
                found = True

            # 3. 종목명으로 검색 (Naver 크롤링 데이터인 경우 Ticker가 없을 수 있음)
            if not found:
                # 문자열 컬럼들 중에서 종목명이 포함된 행 찾기
                # search_query: "삼성전자"
                for col in pdf_df.columns:
                    # 데이터 타입이 문자열이거나 object인 경우
                    if pdf_df[col].dtype == object or pdf_df[col].dtype == str:
                        # 정확히 일치하거나 포함되는지 확인 (여기선 정확 일치 선호하나, 공백 이슈 등으로 포함 사용)
                        # 하지만 "삼성" 검색 시 "삼성전자"가 걸리는건 의도된 동작.
                        # "삼성전자" 검색 시 "삼성전자" 행을 찾아야 함.
                        
                        # 안전한 처리를 위해 string 변환 후 검색
                        matches = pdf_df[pdf_df[col].astype(str).str.contains(search_query, na=False)]
                        if not matches.empty:
                            row = matches.iloc[0]
                            found = True
                            break
            
            if found and row is not None:
                # 컬럼명이 조금씩 다를 수 있으므로 비중 컬럼 찾기
                weight = 0
                
                # 다양한 컬럼명 시도
                cols = pdf_df.columns
                weight_col = next((c for c in cols if '비중' in c), None) # '비중', '비중(%)', '구성비중' 등
                amount_col = next((c for c in cols if '금액' in c or '평가액' in c), None) # '금액', '평가금액'
                
                if weight_col:
                    weight = row[weight_col]
                elif amount_col: 
                    # 금액만 있고 비중 없으면 전체 합 대비 비율 계산
                    # 해당 컬럼의 합
                    try:
                        total_amt = pdf_df[amount_col].sum()
                        if total_amt > 0:
                            weight = (row[amount_col] / total_amt) * 100
                    except:
                        pass
                
                # 비중이 문자열인 경우 처리 (Naver 등)
                if isinstance(weight, str):
                    try:
                        weight = float(weight.replace('%', '').strip())
                    except:
                        pass
                
                result_list.append({
                    "ETF 코드": etf_ticker,
                    "ETF명": data['name'],
                    "종목 비중(%)": weight,
                    "보유 금액": row[amount_col] if amount_col else 0
                })

        # C. 결과 출력
        # C. 결과 출력
        if result_list:
            df_result = pd.DataFrame(result_list)
            # 비중 내림차순 정렬
            df_result = df_result.sort_values(by="종목 비중(%)", ascending=False).reset_index(drop=True)
            
            st.success(f"총 {len(df_result)}개의 ETF가 해당 종목을 포함하고 있습니다.")
            
            # 테이블
            st.dataframe(
                df_result.style.format({"종목 비중(%)": "{:.2f}", "보유 금액": "{:,.0f}"}),
                use_container_width=True
            )
            
            # 차트 (상위 5개인지, 사용자 선택인지) -> 상위 10개 시각화
            top_n = df_result.head(10)
            fig = px.bar(
                top_n, 
                x="ETF명", 
                y="종목 비중(%)", 
                title=f"'{search_query}' 비중이 높은 ETF Top 10",
                color="종목 비중(%)",
                text="종목 비중(%)"
            )
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            st.warning("해당 종목을 포함하는 ETF가 없습니다.")

# -----------------------------------------------------------------------------
# 1. Helper Functions
# -----------------------------------------------------------------------------
# History functions moved to top

# -----------------------------------------------------------------------------
# 🤖 로보 어드바이저 (Demo) - React Port
# -----------------------------------------------------------------------------
def page_robo_advisor():
    # HTML 공백 제거 헬퍼 함수 (모든 들여쓰기 제거하여 Markdown 코드블록 인식 방지)
    def clean_html(html_str):
        return "\n".join([line.strip() for line in html_str.splitlines() if line.strip()])

    # 모바일 레이아웃을 위한 CSS
    st.markdown(clean_html("""
        <style>
        /* 모바일 화면 시뮬레이션 컨테이너 */
        .mobile-container {
            max-width: 450px;
            margin: 0 auto;
            background-color: #F9FAFB; /* gray-50 */
            min-height: 100vh;
            padding: 20px;
            border-radius: 20px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            font-family: 'Noto Sans KR', sans-serif;
            color: #1F2937; /* gray-800 */
        }
        .stTabs [data-baseweb="tab-list"] {
            justify-content: center;
        }
        /* 카드 스타일 유틸리티 */
        .card {
            background-color: white;
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            border: 1px solid #F3F4F6;
            margin-bottom: 12px;
        }
        </style>
    """), unsafe_allow_html=True)

    # -----------------------------
    # 1. Mock Data Definition
    # -----------------------------
    ongoing_changes = [
        {"id": 1, "name": "KCGI샐러리맨증권자투자신탁(주식)", "type": "out", "before": 10.68, "after": 0, "diff": "-10.68%", "region": "글로벌", "category": "글로벌주식", "tags": ["#글로벌가치주", "#지배구조개선", "#ESG테마"]},
        {"id": 3, "name": "한화천연자원증권자투자신탁(주식)", "type": "new", "before": 0, "after": 4.74, "diff": "+4.74%", "region": "글로벌", "category": "글로벌주식", "tags": ["#에너지/광물", "#천연자원기업", "#실물자산투자"]},
        {"id": 2, "name": "하나PIMCO글로벌인컴혼합자산(채권)", "type": "buy", "before": 2.52, "after": 8.72, "diff": "+6.20%", "region": "글로벌", "category": "선진국채권1", "tags": ["#글로벌채권", "#월지급식", "#안정적수익"]},
        {"id": 4, "name": "교보악사파워인덱스(주식)", "type": "new", "before": 0, "after": 2.88, "diff": "+2.88%", "region": "한국", "category": "국내주식", "tags": ["#KOSPI200", "#국내대형주", "#지수추종"]},
        {"id": 5, "name": "키움슈로더이머징위너스(주식혼합)", "type": "sell", "before": 9.07, "after": 7.75, "diff": "-1.32%", "region": "신흥국", "category": "신흥국주식", "tags": ["#신흥국성장주", "#아시아/남미", "#적극운용"]},
        {"id": 6, "name": "미래에셋전략배분TDF2050", "type": "sell", "before": 33.00, "after": 32.63, "diff": "-0.37%", "region": "글로벌", "category": "TDF", "tags": ["#은퇴타겟2050", "#자동자산배분", "#글로벌분산"]}
    ]

    portfolio_profiles = {
        '성장형': {
            "desc": "시장 수익률을 초과하는 고수익을 추구하며, 주식 자산 비중을 가장 높게 가져갑니다.",
            "color": "#DC2626", # bg-red-600
            "riskLevel": 1,
            "items": [
                {"name": "피델리티글로벌테크놀로지증권자투자신탁", "category": "선진국주식", "ratio": 40.0, "tags": ["#글로벌기술주", "#성장주", "#IT섹터"]},
                {"name": "키움슈로더이머징위너스증권자투자신탁", "category": "신흥국주식", "ratio": 30.0, "tags": ["#신흥국", "#하이리스크", "#고성장"]},
                {"name": "미래에셋차이나그로스증권자투자신탁", "category": "중국주식", "ratio": 20.0, "tags": ["#중국성장주", "#본토투자", "#소비재"]},
                {"name": "한화천연자원증권자투자신탁", "category": "대체투자", "ratio": 10.0, "tags": ["#원자재", "#변동성", "#인플레이션"]},
            ]
        },
        '성장추구형': {
            "desc": "적극적인 자산 배분을 통해 자산 증식을 목표로 하며, 주식 위주에 채권을 일부 혼합합니다.",
            "color": "#EA580C", # bg-orange-600
            "riskLevel": 2,
            "items": [
                {"name": "미래에셋전략배분TDF2050혼합자산", "category": "TDF", "ratio": 35.0, "tags": ["#은퇴타겟2050", "#주식비중확대", "#글로벌분산"]},
                {"name": "KCGI샐러리맨증권자투자신탁", "category": "글로벌주식", "ratio": 25.0, "tags": ["#글로벌우량주", "#ESG", "#지배구조"]},
                {"name": "키움슈로더이머징위너스증권자투자신탁", "category": "신흥국주식", "ratio": 15.0, "tags": ["#이머징마켓", "#고성장", "#적극운용"]},
                {"name": "삼성미국S&P500인덱스증권자투자신탁", "category": "선진국주식", "ratio": 15.0, "tags": ["#미국지수", "#성장주", "#달러자산"]},
                {"name": "하나PIMCO글로벌인컴혼합자산", "category": "해외채권", "ratio": 10.0, "tags": ["#방어자산", "#채권", "#월지급"]},
            ]
        },
        '위험중립형': {
            "desc": "위험과 수익의 균형을 중시하며, 주식과 채권을 균형 있게 배분합니다.",
            "color": "#2563EB", # bg-blue-600
            "riskLevel": 3,
            "items": [
                {"name": "미래에셋전략배분TDF2035혼합자산", "category": "TDF", "ratio": 40.0, "tags": ["#자산배분", "#글로벌분산", "#중위험"]},
                {"name": "삼성미국S&P500인덱스증권자투자신탁", "category": "선진국주식", "ratio": 20.0, "tags": ["#미국대표지수", "#달러자산", "#대형주"]},
                {"name": "하나PIMCO글로벌인컴혼합자산", "category": "해외채권", "ratio": 20.0, "tags": ["#글로벌채권", "#안정성", "#인컴수익"]},
                {"name": "한화천연자원증권자투자신탁", "category": "대체투자", "ratio": 10.0, "tags": ["#원자재", "#인플레이션헷지", "#실물자산"]},
                {"name": "Plus신종개인용MMF", "category": "유동성", "ratio": 10.0, "tags": ["#유동성관리", "#단기자금", "#수시입출금"]},
            ]
        },
        '안정추구형': {
            "desc": "원금 손실 위험을 낮추면서 시중 금리 +α 수익을 추구합니다.",
            "color": "#0D9488", # bg-teal-600
            "riskLevel": 4,
            "items": [
                {"name": "하나PIMCO글로벌인컴혼합자산", "category": "해외채권", "ratio": 40.0, "tags": ["#글로벌채권", "#월지급", "#안정수익"]},
                {"name": "미래에셋솔로몬중장기국공채", "category": "국내채권", "ratio": 30.0, "tags": ["#국공채", "#중기투자", "#안전자산"]},
                {"name": "미래에셋전략배분TDF2025혼합자산", "category": "TDF", "ratio": 20.0, "tags": ["#보수적배분", "#채권혼합", "#은퇴임박"]},
                {"name": "교보악사파워인덱스증권자투자신탁", "category": "국내주식", "ratio": 10.0, "tags": ["#KOSPI200", "#인덱스", "#시장수익률"]},
            ]
        },
        '안정형': {
            "desc": "예금 수준의 안정성을 추구하며, 단기 채권 및 유동성 자산 위주로 운용합니다.",
            "color": "#16A34A", # bg-green-600
            "riskLevel": 5,
            "items": [
                {"name": "Plus신종개인용MMF", "category": "유동성", "ratio": 60.0, "tags": ["#수시입출금", "#원금보존", "#초단기"]},
                {"name": "우리단기채권증권자투자신탁", "category": "국내채권", "ratio": 30.0, "tags": ["#국공채", "#안정수익", "#단기채"]},
                {"name": "하나PIMCO글로벌인컴혼합자산", "category": "해외채권", "ratio": 10.0, "tags": ["#글로벌채권", "#월지급", "#채권혼합"]},
            ]
        }
    }

    current_holdings = [
        {"id": 101, "name": "미래에셋전략배분TDF2050", "ratio": 32.63, "amount": "40,920,000", "profit": "+15.2%"},
        {"id": 102, "name": "하나PIMCO글로벌인컴혼합자산", "ratio": 8.72, "amount": "10,930,000", "profit": "+3.1%"},
        {"id": 103, "name": "키움슈로더이머징위너스", "ratio": 7.75, "amount": "9,720,000", "profit": "-1.5%"},
        {"id": 104, "name": "한화천연자원증권자투자신탁", "ratio": 4.74, "amount": "5,940,000", "profit": "0.0%"},
        {"id": 105, "name": "교보악사파워인덱스", "ratio": 2.88, "amount": "3,610,000", "profit": "0.0%"},
        {"id": 106, "name": "Plus신종개인용MMF", "ratio": 43.28, "amount": "54,290,000", "profit": "-"},
    ]
    
    # -----------------------------
    # 2. UI Layout (Mobile Frame)
    # -----------------------------
    # 중앙 정렬을 위한 3분할, 가운데 컬럼(col_mobile)만 사용
    # Mobile Width Simulation
    _, col_mobile, _ = st.columns([1, 2, 1])
    
    with col_mobile:
        st.title("🤖 로보 어드바이저")
        
        tab1, tab2 = st.tabs(["📊 오늘의 포트폴리오", "💰 계좌 현황"])

        # TAB 1: Portfolio View
        with tab1:
            col_header, col_hist = st.columns([4, 1])
            with col_header:
                st.caption("기준일: 2026.01.14")
                st.subheader("AI 추천 포트폴리오 ✅")
            with col_hist:
                st.button("📜", key="history_btn", help="이력 보기")

            # A. Investment Profile Selection
            profile_names = list(portfolio_profiles.keys())
            selected_profile = st.radio("투자 성향 선택", profile_names, index=0, horizontal=True)
            
            profile = portfolio_profiles[selected_profile]
            is_my_profile = (selected_profile == '성장형')
            
            # HTML flattened with clean_html
            profile_html = clean_html(f"""
            <div style="background-color: {profile['color']}; padding: 24px; border-radius: 20px; color: white; margin-bottom: 24px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); position: relative; overflow: hidden;">
                <div style="position: absolute; right: -16px; top: -16px; background-color: rgba(255,255,255,0.1); width: 96px; height: 96px; border-radius: 50%; filter: blur(24px);"></div>
                <div style="position: relative; z-index: 10;">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;">
                        <div style="display: flex; align-items: center; gap: 4px;">
                            <span style="background-color: rgba(255,255,255,0.2); backdrop-filter: blur(4px); padding: 4px 8px; border-radius: 6px; font-size: 11px; font-weight: bold; display: inline-flex; align-items: center;">
                                🎯 위험등급 {profile['riskLevel']}등급
                            </span>
                        </div>
                        {'<span style="background-color: white; color: #DC2626; padding: 4px 8px; border-radius: 999px; font-size: 10px; font-weight: bold; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">나의 투자성향</span>' if is_my_profile else ''}
                    </div>
                    <h3 style="margin: 0 0 4px 0; font-size: 20px; font-weight: 800; color:white;">{selected_profile} 전략</h3>
                    <p style="margin: 0; font-size: 13px; opacity: 0.9; color: rgba(255,255,255,0.9); line-height: 1.5;">{profile['desc']}</p>
                </div>
            </div>
            """)
            st.markdown(profile_html, unsafe_allow_html=True)

            # B. Items List
            st.markdown(f"**구성 상품 ({len(profile['items'])}개)**")
            
            for item in profile['items']:
                # Card Style Container
                with st.container():
                    st.markdown(clean_html(f"""
                    <div class="card" style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1; padding-right: 16px;">
                            <div style="display: flex; gap: 4px; margin-bottom: 4px;">
                                <span style="font-size: 10px; background-color: #F3F4F6; color: #6B7280; padding: 2px 6px; border-radius: 4px; font-weight: 500;">{item['category']}</span>
                            </div>
                            <h4 style="margin: 0 0 8px 0; font-size: 14px; font-weight: bold; color: #111827; line-height: 1.3;">{item['name']}</h4>
                            <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                                {''.join([f'<span style="font-size: 10px; color: #9CA3AF; background-color: #F9FAFB; padding: 2px 6px; border-radius: 4px;">{tag}</span>' for tag in item['tags']])}
                            </div>
                        </div>
                        <div style="width: 50px; height: 50px; background-color: #F9FAFB; border-radius: 12px; display: flex; flex-direction: column; justify-content: center; align-items: center; border: 1px solid #E5E7EB;">
                            <span style="font-size: 16px; font-weight: bold; color: #1F2937;">{item['ratio']}<span style="font-size: 10px;">%</span></span>
                            <span style="font-size: 9px; color: #9CA3AF;">비중</span>
                        </div>
                    </div>
                    """), unsafe_allow_html=True)
            
            st.button(f"{selected_profile}으로 변경 예약하기", use_container_width=True, type="primary")
            st.caption("* 변경 예약 시 다음 리밸런싱 주기에 반영됩니다.", help="매월 말일 기준")

        # TAB 2: Status View
        with tab2:
            is_rebalancing = st.toggle("리밸런싱 진행중 (Demo)", value=True)
            
            # Account Header
            st.markdown(clean_html("""
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 16px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
                        <span style="background-color: #F3F4F6; color: #4B5563; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: bold;">연금저축</span>
                        <span style="font-size: 13px; font-weight: bold; color: #1F2937; text-decoration: underline; text-decoration-color: #D1D5DB; text-underline-offset: 4px;">123-45-678910</span>
                    </div>
                    <h2 style="margin: 0; font-size: 18px; font-weight: 800; color: #111827;">Global Quants EMP</h2>
                </div>
            </div>
            """), unsafe_allow_html=True)
            
            # Dark Card (Performance) with Gradient
            st.markdown(clean_html("""
            <div style="background: linear-gradient(135deg, #111827, #1F2937); color: white; padding: 20px; border-radius: 20px; margin-bottom: 24px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.2); position: relative; overflow: hidden;">
                <div style="position: absolute; right: -20px; top: -20px; background-color: rgba(255,255,255,0.05); width: 128px; height: 128px; border-radius: 50%; filter: blur(30px);"></div>
                <div style="position: relative; z-index: 10;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 16px;">
                        <div>
                            <div style="color: #9CA3AF; font-size: 12px; margin-bottom: 4px; font-weight: 500;">총 평가금액</div>
                            <div style="font-size: 24px; font-weight: 800; letter-spacing: -0.5px;">125,430,000원</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="color: #9CA3AF; font-size: 12px; margin-bottom: 4px; font-weight: 500;">누적 수익률</div>
                            <div style="font-size: 20px; font-weight: 800; color: #F87171;">+12.4%</div>
                        </div>
                    </div>
                    <div style="border-top: 1px solid rgba(255,255,255,0.1); padding-top: 16px; display: flex; gap: 12px;">
                        <div style="flex: 1;">
                            <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 6px;">
                                <span style="color: #9CA3AF;">벤치마크 (KOSPI)</span>
                                <span style="color: #E5E7EB; font-weight: bold;">+9.2%</span>
                            </div>
                            <div style="background-color: #374151; height: 6px; border-radius: 999px; overflow: hidden;">
                                <div style="background-color: #9CA3AF; width: 70%; height: 100%;"></div>
                            </div>
                        </div>
                        <div style="width: 1px; background-color: rgba(255,255,255,0.1);"></div>
                        <div style="flex: 1;">
                            <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 6px;">
                                <span style="color: #9CA3AF;">내 포트폴리오</span>
                                <span style="color: #F87171; font-weight: bold;">+12.4%</span>
                            </div>
                            <div style="background-color: #374151; height: 6px; border-radius: 999px; overflow: hidden;">
                                <div style="background-color: #EF4444; width: 90%; height: 100%;"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """), unsafe_allow_html=True)
            
            if is_rebalancing:
                # Rebalancing Status Card
                st.markdown(clean_html("""
                <div style="background-color: #2563EB; border-radius: 16px; padding: 16px; color: white; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);">
                    <div style="display: flex; items-center; gap: 8px; margin-bottom: 8px;">
                        <!-- Spinner Icon simulation -->
                        <span style="font-size: 16px;">🔄</span>
                        <span style="font-weight: bold; font-size: 15px;">리밸런싱 진행 중</span>
                    </div>
                    <p style="font-size: 12px; color: #DBEAFE; margin: 0 0 12px 0; line-height: 1.4;">
                        시장 상황 변화에 맞춰 자산 비중을 '성장형' 모델로 조정하고 있습니다.
                    </p>
                    <div style="background-color: #1E40AF; height: 6px; border-radius: 999px; overflow: hidden; margin-bottom: 6px;">
                        <div style="background-color: white; width: 65%; height: 100%;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 10px; color: #BFDBFE;">
                        <span>매도 완료</span>
                        <span>매수 중 (65%)</span>
                    </div>
                </div>
                """), unsafe_allow_html=True)

                st.markdown(clean_html("""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <h3 style="margin: 0; font-size: 16px; font-weight: bold; color: #111827;">실시간 변경 현황</h3>
                    <span style="background-color: #F3F4F6; color: #6B7280; font-size: 10px; padding: 2px 8px; border-radius: 999px; font-weight: bold;">Live</span>
                </div>
                """), unsafe_allow_html=True)
                
                # Group by category logic
                cats = {}
                for item in ongoing_changes:
                    cat = item['category']
                    if cat not in cats: cats[cat] = []
                    cats[cat].append(item)
                
                sorted_cats = sorted(cats.keys(), key=lambda k: len(cats[k]), reverse=True)
                
                for cat in sorted_cats:
                    # Category Header
                    st.markdown(clean_html(f"""
                    <div style="background-color: #F9FAFB; padding: 10px 16px; border-top-left-radius: 12px; border-top-right-radius: 12px; border: 1px solid #F3F4F6; border-bottom: none; margin-top: 12px;">
                        <span style="font-size: 12px; font-weight: bold; color: #374151;">{cat}</span>
                    </div>
                    """), unsafe_allow_html=True)
                    
                    # Items
                    for idx, item in enumerate(cats[cat]):
                        is_last = (idx == len(cats[cat]) - 1)
                        border_style = "border-bottom-left-radius: 12px; border-bottom-right-radius: 12px;" if is_last else ""
                        
                        # Type Badge Colors
                        type_colors = {
                            "new": ("#FEF2F2", "#DC2626", "신규편입"), # bg-red-50, text-red-600
                            "out": ("#EFF6FF", "#2563EB", "전량매도"), # bg-blue-50, text-blue-600
                            "buy": ("#FEF2F2", "#DC2626", "비중확대"),
                            "sell": ("#EFF6FF", "#2563EB", "비중축소")
                        }
                        bg_c, text_c, label = type_colors.get(item['type'], ("#F3F4F6", "#6B7280", item['type']))
                        
                        diff_color = "#DC2626" if item['diff'].startswith('+') else "#2563EB"
                        
                        st.markdown(clean_html(f"""
                        <div style="background-color: white; padding: 16px; border: 1px solid #F3F4F6; {border_style} touch-action: manipulation;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                                <div>
                                    <div style="margin-bottom: 4px;">
                                        <span style="background-color: {bg_c}; color: {text_c}; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: bold;">{label}</span>
                                    </div>
                                    <h4 style="margin: 0; font-size: 14px; font-weight: bold; color: #1F2937; margin-bottom: 4px;">{item['name']}</h4>
                                    <div style="display: flex; gap: 4px;">
                                        {''.join([f'<span style="font-size: 10px; color: #9CA3AF; border: 1px solid #F3F4F6; padding: 2px 6px; border-radius: 4px;">{tag}</span>' for tag in item['tags']])}
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Visualization Before -> After -->
                            <div style="display: flex; align-items: center; gap: 12px; background-color: #F9FAFB; padding: 10px; border-radius: 8px;">
                                <div style="text-align: center; width: 40px;">
                                    <div style="font-size: 9px; color: #9CA3AF; margin-bottom: 2px;">이전</div>
                                    <div style="font-size: 11px; font-weight: 500; color: #6B7280;">{item['before']}%</div>
                                </div>
                                <div style="flex: 1; display: flex; align-items: center; gap: 4px;">
                                    <div style="flex: 1; height: 1px; background-color: {bg_c};"></div>
                                    <div style="background-color: {bg_c}; color: {text_c}; border: 1px solid {bg_c}; padding: 2px 8px; border-radius: 999px; font-size: 10px; font-weight: bold;">
                                        {item['diff']}
                                    </div>
                                    <div style="flex: 1; height: 1px; background-color: {bg_c};"></div>
                                </div>
                                <div style="text-align: center; width: 40px;">
                                    <div style="font-size: 9px; color: #9CA3AF; margin-bottom: 2px;">현재</div>
                                    <div style="font-size: 11px; font-weight: bold; color: {text_c};">{item['after']}%</div>
                                </div>
                            </div>
                        </div>
                        """), unsafe_allow_html=True)
            
            else:
                st.subheader("현재 보유 자산")
                for holding in current_holdings:
                    profit_color = "#DC2626" if holding['profit'].startswith('+') else ("#2563EB" if holding['profit'].startswith('-') else "#6B7280")
                    st.markdown(clean_html(f"""
                    <div class="card" style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4 style="margin: 0 0 4px 0; font-size: 14px; font-weight: bold; color: #1F2937;">{holding['name']}</h4>
                            <div style="display: flex; align-items: center; gap: 6px;">
                                <span style="font-size: 12px; color: #6B7280;">{holding['amount']}원</span>
                                <span style="font-size: 10px; background-color: #F3F4F6; color: #6B7280; padding: 2px 6px; border-radius: 4px;">{holding['ratio']}%</span>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 14px; font-weight: bold; color: {profit_color};">{holding['profit']}</span>
                        </div>
                    </div>
                    """), unsafe_allow_html=True)

if selection == "🤖 로보 어드바이저 (Demo)":
    page_robo_advisor()

if selection == "🧪 Qlib 실험실 (Pro)":
    st.title("🧪 Qlib 실험실 (Pro)")
    st.caption("Microsoft Qlib 스타일의 전문적인 퀀트 연구 워크플로우를 제공합니다. Factor IC 분석을 통해 알파를 검증하세요.")
    
    # 1. 사이드바 설정 (Dataset & Model)
    with st.sidebar:
        st.header("🔬 실험 설정 (Experiment Config)")
        
        # Universe Reuse
        universe_preset = st.selectbox(
            "유니버스 선택", 
            ["NASDAQ Top 10 (Demo)", "Tech Giants (M7)", "NASDAQ Top 30", "직접 입력"]
        )
        
        if universe_preset == "직접 입력":
            tickers_input = st.text_input("종목 코드 (쉼표 구분)", "AAPL, MSFT, GOOGL")
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        elif universe_preset == "Tech Giants (M7)":
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
        elif universe_preset == "NASDAQ Top 10 (Demo)":
            tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "COST", "PEP"]
        else: # NASDAQ Top 30
            # Sample subset
            tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "COST", "PEP", "CSCO", "NFLX", "AMD"]

        st.divider()
        start_date = st.date_input("데이터 시작일", pd.to_datetime("2020-01-01"))
        split_date = st.date_input("학습/테스트 분할일", pd.to_datetime("2023-01-01"))
        
        st.divider()
        st.subheader("⚙️ 모델 & 팩터 설정 (Pro)")
        
        # Factor Groups (Mock for now, will filter later if needed)
        factor_groups = st.multiselect(
            "활성화할 팩터 그룹",
            ["Price Momentum (ROC, KMID)", "Volatility (Std, ATR)", "Volume (VMA)"],
            default=["Price Momentum (ROC, KMID)", "Volatility (Std, ATR)", "Volume (VMA)"]
        )
        
        st.caption("LightGBM 하이퍼파라미터")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            lgbm_leaves = st.slider("Num Leaves", 10, 255, 31)
            lgbm_depth = st.slider("Max Depth", -1, 20, -1)
        with col_p2:
            lgbm_lr = st.select_slider("Learning Rate", options=[0.001, 0.005, 0.01, 0.05, 0.1], value=0.05)
            lgbm_min_data = st.slider("Min Data Leaf", 10, 100, 20)
            
        with st.expander("고급 설정 (Advanced)"):
            lgbm_feature_frac = st.slider("Feature Fraction", 0.5, 1.0, 0.8, help="트리 생성 시 무작위로 선택할 Feature 비율")
            lgbm_bagging_frac = st.slider("Bagging Fraction", 0.5, 1.0, 0.8, help="데이터 샘플링 비율")
        
    # 2. Main Workspace
    st.info("💡 **Qlib Workflow**: Data Loader -> Alpha Factory (Feature Eng.) -> Label Gen -> LightGBM -> IC Analysis")
    
    if st.button("🚀 실험 시작 (Run Experiment)"):
        import qlib_workflow
        import importlib
        importlib.reload(qlib_workflow) # Reload for dev
        from qlib_workflow import QlibWorkflow
        
        # 2.1 Data Loading (Inline OR Lambda)
        def my_loader(tickers, start, end):
            # Wrapper around yf.download
            # Note: We need a buffer for features (e.g. 60 days)
            actual_start = pd.to_datetime(start) - pd.Timedelta(days=100)
            with st.spinner(f"데이터 다운로드 중... ({len(tickers)} 종목)"):
                try:
                    df = yf.download(tickers, start=actual_start, end=end, group_by='ticker', progress=False)
                    # Convert MultiIndex Columns to Dict of DFs
                    res = {}
                    if len(tickers) == 1:
                        res[tickers[0]] = df
                    else:
                        for t in tickers:
                            try:
                                res[t] = df[t].dropna()
                            except: pass
                    return res
                except Exception as e:
                    st.error(f"Download Error: {e}")
                    return {}

        qc = QlibWorkflow(my_loader)
        
        # 2.2 Orchestration
        with st.status("🔬 퀀트 파이프라인 가동 중...", expanded=True) as status:
            st.write("1️⃣ 데이터 준비 및 알파 생성 중...")
            dataset, err = qc.prepare_data(tickers, start_date, pd.to_datetime("today"))
            
            if err or dataset is None:
                st.error(f"데이터 준비 실패: {err}")
                st.stop()
                
            st.success(f"데이터 준비 완료! (Shape: {dataset.shape})")
            st.write(f"📊 생성된 팩터(Features): {len(dataset.columns)-1}개")
            st.dataframe(dataset.head(5))
            
            # Split
            st.write("2️⃣ 학습/테스트 데이터 분할 중...")
            dates = dataset.index.get_level_values('Date')
            train_mask = dates < pd.to_datetime(split_date)
            test_mask = dates >= pd.to_datetime(split_date)
            
            train_df = dataset[train_mask]
            test_df = dataset[test_mask]
            
            # Feature Cols (All except Ref($close, -1))
            label_col = [c for c in dataset.columns if "Ref" in c][0]
            feature_cols = [c for c in dataset.columns if c != label_col]
            
            st.write(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
            
            st.write("3️⃣ LightGBM 모델 학습 중...")
            
            # Pass Params
            lgbm_params = {
                'num_leaves': lgbm_leaves,
                'learning_rate': lgbm_lr,
                'max_depth': lgbm_depth,
                'min_child_samples': lgbm_min_data,
                'colsample_bytree': lgbm_feature_frac,
                'subsample': lgbm_bagging_frac,
                'n_estimators': 300
            }
            
            model = qc.train_model(train_df, test_df, feature_cols, label_col, **lgbm_params) 
            st.success("모델 학습 완료!")
            
            st.write("4️⃣ 성과 분석 (IC Analysis) 중...")
            metrics, daily_ic, daily_rank_ic, res_df = qc.analyze_performance(model, test_df, feature_cols, label_col)
            
            status.update(label="✅ 실험 완료!", state="complete", expanded=False)
            
        # 3. Report
        st.divider()
        st.header("📊 실험 결과 리포트 (Experiment Report)")
        
        # Metrics Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("IC (Mean)", f"{metrics['IC_Mean']:.4f}")
        c2.metric("ICIR", f"{metrics['ICIR']:.4f}")
        c3.metric("Rank IC (Mean)", f"{metrics['Rank_IC_Mean']:.4f}")
        c4.metric("Rank ICIR", f"{metrics['Rank_ICIR']:.4f}")
        
        st.caption("""
        * **IC (Information Coefficient)**: 예측과 실제 수익률의 상관계수. (0.05 이상임 훌륭)
        * **ICIR**: IC의 안정성 (Mean / Std). 높을수록 꾸준한 예측력.
        """)
        
        # Charts
        tab1, tab2, tab3 = st.tabs(["📉 누적 IC (Cumulative IC)", "🔍 Feature Importance", "📈 예측 vs 실제"])
        
        with tab1:
            st.subheader("일별 Rank IC 추이")
            cum_rank_ic = daily_rank_ic.cumsum()
            st.line_chart(cum_rank_ic)
            st.caption("우상향할수록 모델이 꾸준히 시장을 맞추고 있음을 의미합니다.")
            
        with tab2:
            st.subheader("주요 팩터 중요도 (Top 10)")
            imp = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False).head(10)
            
            import plotly.express as px
            fig = px.bar(imp, x='Importance', y='Feature', orientation='h', title="Feature Importance")
            st.plotly_chart(fig)
            
        with tab3:
            st.subheader("테스트 기간 예측 분포")
            st.scatter_chart(res_df.reset_index(), x='Pred', y=label_col, color='#00CC96')
