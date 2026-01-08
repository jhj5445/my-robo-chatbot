import streamlit as st
import google.generativeai as genai
import os
import glob
import re
import streamlit.components.v1 as components
import yfinance as yf
import plotly.express as px
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import numpy as np
import scipy.optimize as sco


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
    st.title("메뉴")
    selection = st.radio("이동할 페이지를 선택하세요:", ["🤖 챗봇", "📄 Macro Takling Point", "📈 전략 실험실 (Beta)", "🤖 AI 모델 테스팅", "⚖️ 포트폴리오 최적화", "🔍 기술적 패턴 스캐너"], label_visibility="collapsed")

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

if selection == "📄 Macro Takling Point":
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
                ["직접 입력", "NASDAQ Top 10 (Demo)", "Tech Giants (M7)", "NASDAQ Top 30 (Big Tech)", "S&P 500 Top 50 (Sector Leaders)"]
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

            if universe_preset != "직접 입력":
                st.info(f"선택된 유니버스: {len(tickers)}개 종목")

        with col2:
            model_type = st.selectbox("사용할 AI 모델", ["Linear Regression (선형회귀)", "LightGBM (트리 부스팅)", "SVM (Support Vector Machine)"])
            
            # Feature 복잡도 선택
            feature_level = st.radio(
                "Feature 복잡도 (AI 지능)", 
                ["Light (5개 - 속도 중심)", "Standard (22개 - 균형)", "Rich (50+개 - 정밀 분석)"],
                index=1
            )
            
            # Top-K 선택
            top_k_select = st.number_input("일일 매수 종목 수 (Top K)", min_value=1, max_value=10, value=3)
    
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
                
                # ---------------- [Feature Engineering] ----------------
                feature_cols = []
                
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
                    # 공통: 체계적 Feature 생성 (Windows Loop)
                    
                    # Windows 설정
                    if "Rich" in feature_level:
                        windows = [3, 5, 10, 20, 40, 60, 120] # Rich: 초단기(3) 및 초장기(120) 추가
                    else:
                        windows = [5, 10, 20, 60] # Standard

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
                    
                    # RSI (Standard: 14, 60 / Rich: 9, 14, 28, 60)
                    rsi_windows = [9, 14, 28, 60] if "Rich" in feature_level else [14, 60]
                    for rsi_w in rsi_windows:
                        delta = df['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(rsi_w).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(rsi_w).mean()
                        rs = gain / loss
                        col_rsi = f'RSI_{rsi_w}'
                        df[col_rsi] = 100 - (100 / (1 + rs))
                        feature_cols.append(col_rsi)

                    # [Rich Only Features] 추가
                    if "Rich" in feature_level:
                        # 1. Lagged Returns (시계열 패턴)
                        for lag in [1, 2, 3, 5]:
                            col_lag = f'Ret_Lag_{lag}'
                            df[col_lag] = df['Ret_1d'].shift(lag)
                            feature_cols.append(col_lag)
                        
                        # 2. Candle Patterns
                        # Body Ratio (몸통 길이 / 전체 길이)
                        df['Candle_Body'] = (df['Close'] - df['Open']).abs()
                        df['Candle_Len'] = (df['High'] - df['Low'])
                        df['Body_Ratio'] = df['Candle_Body'] / df['Candle_Len'].replace(0, 1) # Div by zero 방지
                        feature_cols.append('Body_Ratio')
                        
                        # Shadow Upper/Lower
                        df['Shadow_Upper'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Candle_Len'].replace(0, 1)
                        df['Shadow_Lower'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Candle_Len'].replace(0, 1)
                        feature_cols.append('Shadow_Upper')
                        feature_cols.append('Shadow_Lower')
                        
                        # 3. Day of Week (요일 효과)
                        # 원핫 인코딩 대신 간단히 숫자로 (트리 모델은 이거면 충분)
                        df['DayOfWeek'] = df.index.dayofweek
                        feature_cols.append('DayOfWeek')

                # Label (Target): 다음날 수익률
                df['Next_Return'] = df['Close'].pct_change().shift(-1)
                
                df.dropna(inplace=True)
                
                if not df.empty:
                    full_data[ticker] = df
                    valid_tickers.append(ticker)
                    
            except Exception as e:
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
        if "Linear" in model_type:
            model = LinearRegression()
        elif "SVM" in model_type:
            if len(X_train) > 10000:
                st.warning("데이터가 많아 SVM 학습 속도가 느릴 수 있습니다.")
            model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        elif "LightGBM" in model_type:
            model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
            
        model.fit(X_train_scaled, y_train)
        progress_bar.progress(0.7)
        
        # C. 예측 및 백테스팅 (Dynamic Top-K)
        status_text.text(f"백테스팅 시뮬레이션 중 (Top {top_k_select})...")
        
        all_test_dates = sorted(list(set().union(*[d.index for d in test_datasets.values()])))
        
        strategy_capital = 1.0 
        benchmark_capital = 1.0
        portfolio_curve = []
        benchmark_curve = []
        dates = []
        
        current_capital = 1.0
        
        for date in all_test_dates:
            daily_scores = []
            daily_returns = [] 
            
            for ticker in valid_tickers:
                if ticker in test_datasets and date in test_datasets[ticker].index:
                    row = test_datasets[ticker].loc[date]
                    feats = row[feature_cols].values.reshape(1, -1)
                    feats_scaled = scaler.transform(feats)
                    score = model.predict(feats_scaled)[0]
                    daily_scores.append((ticker, score, row['Next_Return']))
                    daily_returns.append(row['Next_Return'])
            
            if not daily_scores:
                continue
                
            # Benchmark
            avg_daily_ret = np.mean(daily_returns)
            benchmark_capital *= (1 + avg_daily_ret)
            
            # Strategy: User Selected Top-K
            daily_scores.sort(key=lambda x: x[1], reverse=True) 
            
            # 입력된 k보다 유효 종목이 적으면 가능한 만큼만 매수
            actual_k = min(top_k_select, len(daily_scores))
            selected = daily_scores[:actual_k]
            
            if selected:
                strategy_daily_ret = np.mean([x[2] for x in selected])
            else:
                strategy_daily_ret = 0.0
                
            strategy_capital *= (1 + strategy_daily_ret)
            
            portfolio_curve.append(strategy_capital)
            benchmark_curve.append(benchmark_capital)
            dates.append(date)
            
        progress_bar.progress(1.0)
        status_text.empty()
        
        # D. 결과 저장 (Session State)
        st.session_state.trained_models[model_type] = {
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "full_data": full_data,
            "valid_tickers": valid_tickers,
            "top_k": top_k_select,   # 저장: 학습할 때 쓴 Top-K
            "feature_level": feature_level # 저장: 학습할 때 쓴 레벨
        }
        
        # E. 결과 시각화
        results_df = pd.DataFrame({
            "Date": dates,
            "AI Model Portfolio": portfolio_curve,
            "Benchmark (Equal Weight)": benchmark_curve
        }).set_index("Date")
        
        st.success(f"학습 완료! ({model_type}) - Features: {len(feature_cols)}개, Top-{top_k_select}")
        
        total_ret = results_df['AI Model Portfolio'].iloc[-1] - 1
        bench_ret = results_df['Benchmark (Equal Weight)'].iloc[-1] - 1
        alpha = total_ret - bench_ret
        
        c1, c2, c3 = st.columns(3)
        c1.metric("AI 포트폴리오 수익률", f"{total_ret:.2%}", delta=f"{alpha:.2%}")
        c2.metric("벤치마크 수익률", f"{bench_ret:.2%}")
        mdd_series = results_df['AI Model Portfolio'] / results_df['AI Model Portfolio'].cummax() - 1
        mdd = mdd_series.min()
        c3.metric("최대 낙폭 (MDD)", f"{mdd:.2%}")
        
        st.subheader(f"📈 백테스팅 결과: AI Top-{top_k_select} 전략 vs 시장")
        fig = px.line(results_df, title=f"{model_type} 기반 Top-{top_k_select} 전략 성과")
        st.plotly_chart(fig, use_container_width=True)
        
        if "Linear" in model_type or "LightGBM" in model_type:
            st.subheader(f"🔍 모델 중요 Feature (Top 20 / {len(feature_cols)})")
            if "Linear" in model_type:
                importance = np.abs(model.coef_)
            else:
                importance = model.feature_importances_
            
            imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": importance}).sort_values(by="Importance", ascending=False)
            st.bar_chart(imp_df.head(20).set_index("Feature"))

    # F. 오늘의 추천 PICK (별도 섹션)
    st.divider()
    
    if not st.session_state.trained_models:
        st.subheader("🔮 오늘의 추천 PICK")
        st.info("👆 위에서 먼저 AI 모델을 학습시켜주세요.")
    else:
        # 학습된 모델 선택
        model_options = list(st.session_state.trained_models.keys())
        selected_model_name = st.selectbox("추천을 확인할 학습 모델 선택", model_options)
        
        # 저장된 모델 정보 로드
        saved_info = st.session_state.trained_models[selected_model_name]
        saved_top_k = saved_info.get("top_k", 3)
        
        st.subheader(f"🔮 오늘의 추천 PICK (Daily Top {saved_top_k})")

        # 캐시 키 생성 (날짜 + 모델명 + TopK)
        today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
        cache_key = f"{selected_model_name}_{today_str}_{saved_top_k}"
        
        # 이미 분석한 결과가 있는지 확인
        if cache_key in st.session_state.gemini_insights:
            st.success(f"⚡ 저장된 분석 결과 (Date: {today_str})")
            cached_data = st.session_state.gemini_insights[cache_key]
            
            # 카드 표시 (Top K 개수만큼 컬럼 동적 생성 - 너무 많으면 3개씩)
            st.write(f"**추천 종목 ({len(cached_data['top_k_items'])}개)**")
            
            cols = st.columns(min(len(cached_data['top_k_items']), 4)) # 최대 4열
            for i, item in enumerate(cached_data['top_k_items']):
                col_idx = i % 4
                with cols[col_idx]:
                    st.info(f"**{i+1}위: {item['Ticker']}**\n\nAI Score: {item['Score']:.4f}")
            
            st.markdown(cached_data['insight'])
            
        else:
            if st.button("🚀 추천 종목 분석 실행 (Gemini)"):
                with st.spinner(f"최신 데이터 분석 중... (Top {saved_top_k})"):
                    model = saved_info['model']
                    scaler = saved_info['scaler']
                    feature_cols = saved_info['feature_cols']
                    full_data = saved_info['full_data']
                    valid_tickers = saved_info['valid_tickers']
                    
                    today_scores = []
                    
                    for ticker in valid_tickers:
                        try:
                            df = full_data[ticker]
                            last_row = df.iloc[[-1]] 
                            last_date = last_row.index[0].strftime('%Y-%m-%d')
                            
                            feats = last_row[feature_cols].values
                            feats_scaled = scaler.transform(feats)
                            score = model.predict(feats_scaled)[0]
                            
                            # 대표 Feature 값 추출 (설명을 위해 일부만)
                            # 간단히 첫 5개나 주요 feature 이름 매칭해서 보낼 수 있음
                            feat_dict = {}
                            # Common features across levels
                            if "RSI_14" in last_row.columns: feat_dict["RSI_14"] = f"{last_row['RSI_14'].values[0]:.2f}"
                            elif "RSI" in last_row.columns: feat_dict["RSI"] = f"{last_row['RSI'].values[0]:.2f}" # For Light mode
                            
                            if "ROC_20" in last_row.columns: feat_dict["ROC_20 (Momentum)"] = f"{last_row['ROC_20'].values[0]:.2%}"
                            elif "Momentum_1M" in last_row.columns: feat_dict["Momentum_1M"] = f"{last_row['Momentum_1M'].values[0]:.2%}" # For Light mode
                            
                            if "MA_Dist_20" in last_row.columns: feat_dict["MA_Dist_20"] = f"{last_row['MA_Dist_20'].values[0]:.4f}"
                            elif "Disparity_20" in last_row.columns: feat_dict["Disparity_20"] = f"{last_row['Disparity_20'].values[0]:.4f}" # For Light mode
                            
                            if "Vol_20" in last_row.columns: feat_dict["Vol_20"] = f"{last_row['Vol_20'].values[0]:.4f}"
                            elif "Volatility" in last_row.columns: feat_dict["Volatility"] = f"{last_row['Volatility'].values[0]:.4f}" # For Light mode
                            
                            if not feat_dict: # Rich 모드 등으로 이름이 다를 경우 대비 안전장치
                                feat_dict = {"Score": f"{score:.4f}"}

                            today_scores.append({
                                "Ticker": ticker,
                                "Score": score,
                                "Date": last_date,
                                "Features": feat_dict
                            })
                        except Exception as e:
                            # st.warning(f"Error processing {ticker} for daily pick: {e}")
                            pass
                    
                    # Top K 선정
                    today_scores.sort(key=lambda x: x['Score'], reverse=True)
                    top_k_items = today_scores[:saved_top_k]
                    
                    if top_k_items:
                        # Gemini 프롬프트
                        prompt_context = f"Model Type: {selected_model_name}\nTarget Strategy: Buy Top {saved_top_k} scores daily.\n\nTop {saved_top_k} Recommended Stocks:\n"
                        for i, item in enumerate(top_k_items):
                            prompt_context += f"{i+1}. {item['Ticker']} (Score: {item['Score']:.4f})\n   - Indicators: {item['Features']}\n"
                        prompt_context += "\nAct as a Quantitative Analyst. Explain WHY the model likely selected these stocks based on the provided indicators. Focus on the quantitative rationale. Write in Korean."
                        
                        try:
                            # API Key Rotation 적용
                            insight_text = generate_content_with_rotation(prompt_context, model_name="gemini-3-flash-preview")
                            
                            # 결과 캐싱
                            st.session_state.gemini_insights[cache_key] = {
                                "top_k_items": top_k_items,
                                "insight": insight_text
                            }
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Gemini 분석 중 오류: {e}")
                    else:
                        st.warning("데이터 부족으로 예측할 수 없습니다.")

elif selection == "⚖️ 포트폴리오 최적화":
    st.title("⚖️ 포트폴리오 최적화 (Portfolio Optimizer)")
    st.caption("현대 포트폴리오 이론(MPT)에 기반하여 최적의 자산 배분 비율을 제안합니다.")

    # 1. 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 최적화 설정")
        tickers_string = st.text_area(
            "포트폴리오 구성 종목 (쉼표 구분)", 
            value="AAPL, MSFT, GOOGL, AMZN, TSLA, SPY, GLD, TLT",
            height=100
        )
        
        col1, col2 = st.columns(2)
        with col1:
            start_date_opt = st.date_input("분석 시작일", pd.to_datetime("2020-01-01"))
        with col2:
            end_date_opt = st.date_input("분석 종료일", pd.to_datetime("today"))
            
        risk_free_rate = st.number_input("무위험 이자율 (%)", value=3.5, step=0.1) / 100

    # 2. 데이터 다운로드 및 처리
    st.info("💡 **Efficient Frontier (효율적 투자선)**: 동일한 위험 수준에서 최대 수익을 내거나, 동일한 기대 수익에서 최소 위험을 갖는 포트폴리오의 집합입니다.")
    
    if st.button("🚀 포트폴리오 최적화 실행"):
        tickers = [t.strip().upper() for t in tickers_string.split(',') if t.strip()]
        
        if len(tickers) < 2:
            st.warning("최소 2개 이상의 종목을 입력해주세요.")
            st.stop()
            
        with st.spinner("데이터 수집 및 최적화 계산 중..."):
            # 데이터 수집
            data = pd.DataFrame()
            valid_tickers = []
            
            for t in tickers:
                try:
                    df = yf.download(t, start=start_date_opt, end=end_date_opt, progress=False)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                        
                    if 'Adj Close' in df.columns:
                        series = df['Adj Close']
                    elif 'Close' in df.columns:
                        series = df['Close']
                    else:
                        continue
                        
                    data[t] = series
                    valid_tickers.append(t)
                except Exception as e:
                    pass
            
            if data.empty or len(valid_tickers) < 2:
                st.error("유효한 데이터가 충분하지 않습니다. 종목 코드를 확인해주세요.")
                st.stop()
                
            # 결측치 처리
            data = data.dropna()
            
            # 수익률 계산
            returns = data.pct_change().dropna()
            mean_returns = returns.mean() * 252 # 연간 기대 수익률
            cov_matrix = returns.cov() * 252 # 연간 공분산
            
            # ---------------------------------------------------------
            # 포트폴리오 최적화 함수 (Scipy)
            # ---------------------------------------------------------
            def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
                returns = np.sum(mean_returns * weights)
                std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return std, returns

            def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
                p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
                return -(p_ret - risk_free_rate) / p_var

            # 제약 조건
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0.0, 1.0) for asset in range(len(valid_tickers)))
            
            # 초기값 (균등 배분)
            num_assets = len(valid_tickers)
            init_guess = num_assets * [1. / num_assets,]
            
            # 최적화 실행
            opt_result = sco.minimize(
                neg_sharpe_ratio, 
                init_guess, 
                args=(mean_returns, cov_matrix, risk_free_rate), 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints
            )
            
            max_sharpe_weights = opt_result['x']
            max_sharpe_std, max_sharpe_ret = portfolio_annualised_performance(max_sharpe_weights, mean_returns, cov_matrix)
            max_sharpe_sharpe = (max_sharpe_ret - risk_free_rate) / max_sharpe_std
            
            # ---------------------------------------------------------
            # 결과 시각화
            # ---------------------------------------------------------
            
            # 1. 파이 차트 (최적 비중)
            st.divider()
            
            weights_df = pd.DataFrame({
                "Ticker": valid_tickers,
                "Weight": max_sharpe_weights
            })
            weights_df = weights_df[weights_df['Weight'] > 0.0001] # 0% 제외
            weights_df['Weight_Pct'] = (weights_df['Weight'] * 100).round(2)
            weights_df = weights_df.sort_values(by="Weight", ascending=False)
            
            c1, c2 = st.columns([1, 1])
            
            with c1:
                st.subheader("🎯 최적 포트폴리오 비중")
                st.caption(f"Max Sharpe Ratio: {max_sharpe_sharpe:.4f}")
                
                fig_pie = px.pie(
                    weights_df, 
                    values='Weight', 
                    names='Ticker', 
                    title='Optimal Asset Allocation',
                    hole=0.4
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with c2:
                st.subheader("📊 예상 성과 (연간)")
                st.metric("기대 수익률 (Annual Return)", f"{max_sharpe_ret:.2%}")
                st.metric("변동성 (Annual Volatility)", f"{max_sharpe_std:.2%}")
                st.metric("샤프 비율 (Sharpe Ratio)", f"{max_sharpe_sharpe:.4f}")
                
                st.markdown("#### 보유 비중 상세")
                st.dataframe(weights_df[['Ticker', 'Weight_Pct']].style.format({"Weight_Pct": "{:.2f}%"}), hide_index=True)

            # 2. 효율적 투자선 차트 (시뮬레이션)
            st.subheader("📈 효율적 투자선 (Efficient Frontier)")
            
            with st.spinner("시뮬레이션 차트 생성 중..."):
                num_portfolios = 5000
                results = np.zeros((3, num_portfolios))
                
                for i in range(num_portfolios):
                    weights = np.random.random(num_assets)
                    weights /= np.sum(weights)
                    
                    p_std, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
                    results[0,i] = p_std
                    results[1,i] = p_ret
                    results[2,i] = (p_ret - risk_free_rate) / p_std
                
                sim_df = pd.DataFrame({
                    "Volatility": results[0,:],
                    "Return": results[1,:],
                    "Sharpe": results[2,:]
                })
                
                fig_ef = px.scatter(
                    sim_df, x="Volatility", y="Return", color="Sharpe",
                    title="Efficient Frontier Simulation (5,000 Portfolios)",
                    color_continuous_scale='Viridis',
                    labels={"Volatility": "리스크 (표준편차)", "Return": "기대 수익률"}
                )
                
                # 최적점 표시
                fig_ef.add_scatter(
                    x=[max_sharpe_std], y=[max_sharpe_ret], 
                    mode='markers+text', 
                    marker=dict(color='red', size=15, symbol='star'),
                    name='Max Sharpe Portfolio',
                    text=['★ Max Sharpe'], textposition="top left"
                )
                
                st.plotly_chart(fig_ef, use_container_width=True)

elif selection == "🔍 기술적 패턴 스캐너":
    st.title("🔍 기술적 패턴 스캐너 (Technical Pattern Scanner)")
    st.caption("전체 시장을 스캔하여 '지금 당장' 의미 있는 차트 패턴이 발생한 종목을 포착합니다.")

    # 1. 스캔 대상 설정
    with st.expander("📡 스캔 설정 (Universe)", expanded=True):
        universe_preset = st.selectbox(
            "스캔 대상 그룹 선택",
            ["NASDAQ Top 30 (Big Tech)", "Dow Jones 30 (Blue Chips)", "S&P 100 (Large Cap)", "직접 입력"]
        )

        if universe_preset == "직접 입력":
            tickers_input = st.text_input("종목 코드 입력 (쉼표 구분)", "AAPL, MSFT, TSLA, NVDA, AMD, INTC, QCOM")
            scan_tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        elif universe_preset == "NASDAQ Top 30 (Big Tech)":
            scan_tickers = [
                "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "COST", "PEP",
                "CSCO", "NFLX", "AMD", "ADBE", "TMUS", "INTC", "QCOM", "TXN", "AMGN", "HON",
                "AMAT", "INTU", "SBUX", "ADP", "BKNG", "GILD", "ISRG", "MDLZ", "REGN", "VRTX"
            ]
        elif universe_preset == "Dow Jones 30 (Blue Chips)":
            scan_tickers = [
                "MMM", "AXP", "AMGN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DIS", 
                "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "MCD", "MRK", 
                "MSFT", "NKE", "PG", "CRM", "TRV", "UNH", "VZ", "V", "WMT", "WBA" # WBA is replaced by AMZN in DJIA recently but keep list simple for now or update
            ]
             # Note: Dow components change. 
        elif universe_preset == "S&P 100 (Large Cap)":
            # Sample list
            scan_tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B", "LLY", "V", "TSM", "UNH", "XOM", "JPM"] 
            st.info("Demo: 속도를 위해 주요 14개 종목만 스캔합니다.")
            
        st.write(f"총 {len(scan_tickers)}개 종목을 분석합니다.")

    # 2. 스캔 실행
    if st.button("🛰️ 패턴 스캔 시작"):
        results = []
        
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        # 데이터 일괄 다운로드 (속도 개선)
        status_text.text("데이터 일괄 다운로드 중...")
        
        # 기간: 넉넉히 120일 (MA60 계산용)
        start_date_scan = pd.to_datetime("today") - pd.Timedelta(days=200)
        
        try:
            # yfinance batch download
            # threads=True is default
            raw_data = yf.download(scan_tickers, start=start_date_scan, group_by='ticker', progress=False)
        except Exception as e:
            st.error(f"데이터 다운로드 실패: {e}")
            st.stop()
            
        status_text.text("패턴 분석 중...")
        
        for i, ticker in enumerate(scan_tickers):
            try:
                # 데이터 추출
                if len(scan_tickers) == 1:
                    df = raw_data
                else:
                    df = raw_data[ticker]
                
                # MultiIndex 컬럼 정리
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # 유효성 검사
                if df.empty or 'Close' not in df.columns:
                    continue
                    
                df = df.dropna(subset=['Close'])
                if len(df) < 60: # 최소 데이터 요구량
                    continue
                
                # ---------------- [패턴 인식 엔진] ----------------
                detected_patterns = []
                
                # 최신 데이터
                curr_price = df['Close'].iloc[-1]
                prev_price = df['Close'].iloc[-2]
                
                # 1. 이평선 (Golden/Death Cross)
                ma20 = df['Close'].rolling(20).mean()
                ma60 = df['Close'].rolling(60).mean()
                
                curr_ma20 = ma20.iloc[-1]
                curr_ma60 = ma60.iloc[-1]
                prev_ma20 = ma20.iloc[-2]
                prev_ma60 = ma60.iloc[-2]
                
                # 골든 크로스: 어제는 20 < 60 이었는데 오늘 20 > 60
                if prev_ma20 < prev_ma60 and curr_ma20 > curr_ma60:
                    detected_patterns.append("✨ Golden Cross (매수 신호)")
                
                # 데드 크로스
                if prev_ma20 > prev_ma60 and curr_ma20 < curr_ma60:
                    detected_patterns.append("💀 Death Cross (매도 신호)")
                    
                # 2. RSI (과매수/과매도)
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                curr_rsi = rsi.iloc[-1]
                
                if curr_rsi < 30:
                    detected_patterns.append(f"🟢 RSI 과매도 ({curr_rsi:.1f}) - 반등 기대")
                elif curr_rsi > 70:
                    detected_patterns.append(f"🔴 RSI 과매수 ({curr_rsi:.1f}) - 조정 주의")
                    
                # 3. 볼린저 밴드 (돌파)
                std = df['Close'].rolling(20).std()
                upper = ma20 + (std * 2)
                lower = ma20 - (std * 2)
                
                curr_upper = upper.iloc[-1]
                curr_lower = lower.iloc[-1]
                
                if curr_price < curr_lower:
                    detected_patterns.append("📉 볼린저 하단 돌파 (과매도)")
                elif curr_price > curr_upper:
                    detected_patterns.append("📈 볼린저 상단 돌파 (강한 상승세)")
                
                # ------------------------------------------------
                
                if detected_patterns:
                    # 결과 저장
                    results.append({
                        "Ticker": ticker,
                        "Price": f"${curr_price:.2f}",
                        "Change": f"{(curr_price - prev_price)/prev_price:.2%}",
                        "Patterns": detected_patterns
                    })
                    
            except Exception as e:
                pass
            
            progress_bar.progress((i + 1) / len(scan_tickers))
            
        status_text.empty()
        progress_bar.empty()
        
        # 결과 출력
        st.divider()
        if results:
            st.success(f"총 {len(results)}개 종목에서 특이 패턴이 발견되었습니다!")
            
            # 보기 좋게 카드 형태로 출력 혹은 데이터프레임
            for item in results:
                with st.container():
                    c1, c2, c3 = st.columns([1, 1.5, 3])
                    c1.subheader(item['Ticker'])
                    c2.metric("현재가", item['Price'], item['Change'])
                    
                    # 뱃지 형태로 패턴 표시
                    with c3:
                        st.write("**발견된 패턴:**")
                        for pat in item['Patterns']:
                            if "매수" in pat or "반등" in pat or "Golden" in pat:
                                st.success(pat)
                            elif "매도" in pat or "주의" in pat or "Death" in pat:
                                st.error(pat)
                            else:
                                st.info(pat)
                    st.divider()
        else:
            st.info("현재 기준 특이 패턴(골든크로스, 과매수/과매도 등)이 발견된 종목이 없습니다.")

