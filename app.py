import streamlit as st
import google.generativeai as genai
import os

# -----------------------------------------------------------------------------
# 1. API 키 설정 (Rotation Logic)
# -----------------------------------------------------------------------------
api_keys = []
if "GOOGLE_API_KEY" in st.secrets:
    api_keys.append(st.secrets["GOOGLE_API_KEY"])
    i = 2
    while f"GOOGLE_API_KEY_{i}" in st.secrets:
        api_keys.append(st.secrets[f"GOOGLE_API_KEY_{i}"])
        i += 1
else:
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        api_keys.append(key)
        i = 2
        while os.getenv(f"GOOGLE_API_KEY_{i}"):
            api_keys.append(os.getenv(f"GOOGLE_API_KEY_{i}"))
            i += 1

if api_keys:
    genai.configure(api_key=api_keys[0])

def generate_content_with_rotation(prompt, model_name="gemini-1.5-flash"):
    if not api_keys:
        raise Exception("API 키가 설정되지 않았습니다.")
    last_error = None
    for i, key in enumerate(api_keys):
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            last_error = e
            error_str = str(e)
            if "429" in error_str or "Quota" in error_str or "Resource exhausted" in error_str:
                if i < len(api_keys) - 1:
                    continue
            break
    raise last_error

# -----------------------------------------------------------------------------
# 2. 시나리오 데이터 (Decision Tree)
# -----------------------------------------------------------------------------
SCENARIO_DATA = {
    "🛠 서비스 기능 및 가입": {
        "가입/설계 방식": "가입과 동시에 맞춤설계가 진행되는 것이 아니며, **고객이 가입 후 직접 맞춤설계를 진행**해야 합니다. 이는 고객에게 포트폴리오를 '추천'드리는 서비스이기 때문입니다.",
        "투자성향 관련": "투자자성향과 상관없이 가입은 가능하며, 최종에는 본인 투자성향에 따른 포트폴리오 유형이 선택되지만 타 유형도 선택 가능합니다. (단, 추가 고지사항 발생)",
        "포트폴리오 수정": "**투자자가 임의로 자산군 비중을 수정하거나 일부 펀드만 교체할 수 없습니다.** 퇴직연금의 위험자산비율 준수를 위해 로보어드바이저가 자동으로 매매를 진행합니다.",
        "펀드 교체 범위": "원칙적으로 보유 중인 공모펀드 전체가 교체 대상입니다. (단, 유사 성능 펀드는 일부 제외될 수 있음)"
    },
    "🚫 가입 불가 요건": {
        "퇴직연금": "MP구독 서비스 이용 계좌 등은 가입이 불가합니다.",
        "개인연금": """
        - 연금개시 정기지급 계좌
        - 대출/정기매도/자동이체 약정 계좌
        - 사고계좌 예 다수
        - 타 자문/일임 서비스 이용 중인 경우
        """,
        "ISA": "계좌해지/이관 신청 중, 자동이체 약정, 사고계좌 등은 가입 불가합니다.",
        "일반계좌/비과세": "신용/대출 약정, 랩/자문 이용 계좌, 고위험 약정 계좌 등은 가입 불가합니다."
    },
    "⚠️ 이용 제한 (MAPIS)": {
        "정상 가입 기준": """
        - 투자자성향: 성장형/성장추구형 (안정형 불가)
        - 투자권유: '희망'
        - 운용가능금액: 1만원 이상
        - 위험자산비율: 70% 이하 (퇴직연금)
        """,
        "제외 펀드": "러시아펀드, 사모펀드, 환매금지/수수료 펀드 등은 운용 대상에서 제외됩니다."
    },
    "📢 최신 업데이트 (2026)": {
        "투자성향 제한": "Q1 '단기생계' 또는 Q2 '원금보존' 선택 시 이용 불가합니다.",
        "투자설명서": "퇴직연금은 설계 직후 1회, 그 외 계좌는 매수 시마다 발송됩니다."
    },
    "📈 매매/수익률 규칙": {
        "매매 불가 시간": "23:55 ~ 24:05 (자정 전후 10분간 주문 불가)",
        "리밸런싱": "수시(비중 차이 발생 시), 정기(40영업일 경과 시) 진행됩니다.",
        "수익률 미노출": "계좌 내 비운용 자산(예금 등)과의 혼동 방지 및 승인 시차로 인해 로보 성과만 별도 확인 필요합니다.",
        "확인 경로": "[MY 로보어드바이저 > 계좌현황] 메뉴에서 확인 가능합니다."
    },
    "🚨 주요 에러 해결": {
        "소수점 매매": "금액이 너무 적으면 펀드 비중이 정수로 계산되지 않아 매도 불가 에러가 발생할 수 있습니다.",
        "위험자산 초과": "당일 매매로 위험자산 70% 초과된 경우, 결제 완료(T+2) 후 다시 시도해주세요.",
        "미국인": "미국 국적자는 가입 불가 펀드가 있어 에러가 발생할 수 있습니다."
    }
}

# -----------------------------------------------------------------------------
# 3. 모델 설정
# -----------------------------------------------------------------------------
faq_text_block = "\n".join([f"{k}: {v}" for k, v in SCENARIO_DATA.items()]) # Simplified for context
system_prompt = f"""
당신은 로보어드바이저 상담 챗봇입니다.
사용자의 질문에 대해 아래 시나리오 데이터를 참고하여 친절하게 답변해주세요.
시나리오 데이터에 없는 내용은 "죄송합니다, 해당 내용은 상담원 연결이 필요합니다."라고 답해주세요.

[데이터]
{faq_text_block}
"""

# -----------------------------------------------------------------------------
# 4. 앱 UI 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="로보어드바이저 시나리오 챗봇", page_icon="🧩", layout="wide")

# CSS 스타일 유지
st.markdown(
    """
    <style>
        .stApp { background-color: #ecf2f5; color: #23292f; }
        [data-testid="stSidebar"] { background-color: #1c2836; color: white; }
        .stButton button { background-color: #ffffff; color: #5383e8; border: 1px solid #5383e8; font-weight: bold; width: 100%; text-align: left; transition: all 0.3s; }
        .stButton button:hover { background-color: #5383e8; color: white; }
        .answer-box { background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #dce2f0; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 20px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧩 로보어드바이저 시나리오 상담")

# API 키 체크
if not api_keys:
    st.error("API 키가 없습니다.")
    st.stop()

# -----------------------------------------------------------------------------
# 5. 상태 관리 (Navigation)
# -----------------------------------------------------------------------------
if 'path' not in st.session_state:
    st.session_state['path'] = [] # 현재 선택 경로 (예: ['서비스 기능', '가입 방식'])

if 'messages' not in st.session_state:
    st.session_state['messages'] = [] # 대화 기록 (AI 채팅용)

def navigate_to(node):
    st.session_state['path'].append(node)

def go_back():
    if st.session_state['path']:
        st.session_state['path'].pop()

def reset_path():
    st.session_state['path'] = []

# -----------------------------------------------------------------------------
# 6. 메인 화면 구성 (2단 분할: 시나리오 선택 / AI 채팅)
# -----------------------------------------------------------------------------
col_scenario, col_chat = st.columns([1, 1])

# [왼쪽] 시나리오 네비게이션
with col_scenario:
    st.subheader("🗂 주제별 찾기")
    
    # 현재 위치 표시
    if st.session_state['path']:
        st.caption(" > ".join(st.session_state['path']))
        if st.button("⬅️ 뒤로가기"):
            go_back()
            st.rerun()
        if st.button("🏠 처음으로"):
            reset_path()
            st.rerun()
    else:
        st.caption("원하는 주제를 선택해주세요.")

    # 현재 데이터 레벨 결정
    current_data = SCENARIO_DATA
    is_leaf = False
    leaf_content = ""

    for step in st.session_state['path']:
        if step in current_data:
            current_data = current_data[step]
            if isinstance(current_data, str):
                is_leaf = True
                leaf_content = current_data
                break
        else:
            # 경로 에러 시 리셋
            reset_path()
            st.rerun()

    # 화면 렌더링
    if is_leaf:
        # 결과 화면
        st.markdown(f"""
        <div class="answer-box">
            <h3>💡 답변</h3>
            {leaf_content}
        </div>
        """, unsafe_allow_html=True)
        st.info("더 궁금한 점이 있으신가요? 챗봇에게 물어보세요 👉")
    else:
        # 버튼 목록 화면
        st.write("")
        for key in current_data.keys():
            if st.button(f"📄 {key}"):
                navigate_to(key)
                st.rerun()

# [오른쪽] AI 챗봇 (Fallback)
with col_chat:
    st.subheader("💬 AI 상담원")
    
    # 채팅 기록 표시
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state['messages']:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # 입력창
    if prompt := st.chat_input("시나리오에 없는 내용은 직접 물어보세요!"):
        # 사용자 메시지 추가
        st.session_state['messages'].append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        
        # AI 답변 생성
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    try:
                        # 현재 시나리오 맥락을 포함할지 여부는 선택사항. 여기선 전체 FAQ 기반.
                        full_prompt = f"질문: {prompt}\n\n답변 (전문가 톤으로):"
                        response = generate_content_with_rotation(full_prompt)
                        st.markdown(response)
                        st.session_state['messages'].append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {e}")
