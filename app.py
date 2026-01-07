import streamlit as st
import google.generativeai as genai


# 1. API 키 설정 (Google AI Studio에서 발급받은 키 입력)
GOOGLE_API_KEY = "AIzaSyBbYUnSBp32fVzTiTlVRcN1GE9JK2BrLKs"
genai.configure(api_key=GOOGLE_API_KEY)

# 2. FAQ 데이터 정의 (여기에 준비하신 FAQ 내용을 자유롭게 넣으세요)
faq_data = """
[로보어드바이저 FAQ]
Q: 로보어드바이저 서비스란 무엇인가요?
A: AI 알고리즘이 고객의 투자 성향에 맞춰 포트폴리오를 자동으로 구성하고 관리해주는 서비스입니다.

Q: 가입 최소 금액은 얼마인가요?
A: 상품별로 다르지만, 보통 10만 원부터 시작 가능합니다.

Q: 수수료는 어떻게 되나요?
A: 연간 운용보수는 약 0.5% 내외이며, 매매 수수료는 별도입니다.
(여기에 더 많은 FAQ 내용을 계속 추가하세요...)
"""

# 3. 모델 설정 (Gemini 3 Flash 사용)
# 시스템 프롬프트에 FAQ 데이터를 주입합니다.
system_prompt = f"""
당신은 '로보어드바이저' 전용 고객상담 AI입니다.
반드시 아래 제공된 [FAQ 데이터]를 바탕으로만 답변하세요.
데이터에 없는 내용이라면 "죄송합니다. 해당 내용은 고객센터(1588-XXXX)로 문의 부탁드립니다"라고 답변하세요.

[FAQ 데이터]
{faq_data}
"""

model = genai.GenerativeModel(
    model_name="gemini-3-flash-preview",
    system_instruction=system_prompt
)

# 4. 웹 화면 UI 구성 (Streamlit)
st.set_page_config(page_title="미래에셋 로보 챗봇", page_icon="🤖")
st.title("🤖 미래에셋 로보어드바이저 상담")
st.caption("FAQ 데이터를 기반으로 AI가 답변해 드립니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 질문 입력
if prompt := st.chat_input("궁금한 점을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 답변 생성
    with st.chat_message("assistant"):
        response = model.generate_content(prompt)
        st.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})
