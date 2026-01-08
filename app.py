import streamlit as st
import google.generativeai as genai
import os
import glob
import re
import streamlit.components.v1 as components


# 1. API 키 설정 (Google AI Studio에서 발급받은 키 입력)
GOOGLE_API_KEY = "AIzaSyBbYUnSBp32fVzTiTlVRcN1GE9JK2BrLKs"
genai.configure(api_key=GOOGLE_API_KEY)

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
st.set_page_config(page_title="미래에셋 로보 챗봇", page_icon="🤖", layout="wide")

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
    selection = st.radio("이동할 페이지를 선택하세요:", ["🤖 챗봇", "📄 Macro Takling Point"], label_visibility="collapsed")

if selection == "🤖 챗봇":
    st.title("🤖 미래에셋 로보어드바이저 상담")
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
                response = model.generate_content(last_user_msg)
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

elif selection == "📄 Macro Takling Point":
    st.title("📄 Macro Talking Point")
    st.caption("각 지수와 날짜별 리포트를 확인하세요.")

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

        # 메인 화면: 리포트 뷰어 (이제 전체 너비 사용)
        if selected_report:
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
