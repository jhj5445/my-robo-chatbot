import streamlit as st
import json
import os
import socket
import re
from datetime import datetime, timedelta, timezone

# 1. KST (한국 표준시 UTC+9) 강제 적용 (Streamlit Cloud 해외 서버 대비)
KST = timezone(timedelta(hours=9))

def get_now():
    return datetime.now(KST)

DATA_FILE = "data.json"

st.set_page_config(
    page_title="우리 아기 이유식 & 큐브 플래너",
    page_icon="🥣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Mobile and Desktop
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 12px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 10px;
    }
    .tag-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    .badge-grain { background-color: #fef3c7; color: #92400e; }
    .badge-protein { background-color: #fee2e2; color: #991b1b; }
    .badge-veg { background-color: #dcfce7; color: #166534; }
    .badge-dday-urgent { background-color: #ef4444; color: white; padding: 2px 8px; border-radius: 6px; font-weight: bold; font-size: 0.85rem;}
    .badge-dday-warn { background-color: #f59e0b; color: white; padding: 2px 8px; border-radius: 6px; font-weight: bold; font-size: 0.85rem;}
    .badge-dday-safe { background-color: #3b82f6; color: white; padding: 2px 8px; border-radius: 6px; font-weight: bold; font-size: 0.85rem;}
    
    .card-urgent { border: 1.5px solid #ef4444; background-color: #fff5f5; border-radius: 10px; padding: 12px; margin-bottom: 8px;}
    .card-warning { border: 1.5px solid #f59e0b; background-color: #fffbeb; border-radius: 10px; padding: 12px; margin-bottom: 8px;}
    .card-good { border: 1.5px solid #10b981; background-color: #f0fdf4; border-radius: 10px; padding: 12px; margin-bottom: 8px;}
    
    .shop-item-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 6px solid #ef4444;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .shop-item-card-warn {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 6px solid #f59e0b;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .mobile-ip-box {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        padding: 10px 16px;
        border-radius: 10px;
        font-size: 0.9rem;
        margin-bottom: 15px;
    }
    .auto-mode-banner {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 15px;
        font-size: 0.92rem;
        color: #1e40af;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# ==========================================
# Google Sheets (Apps Script Web App) & Local Hybrid Storage Manager
# ==========================================
CONFIG_FILE = "config.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_config(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

class StorageManager:
    def __init__(self):
        self.apps_script_url = None
        self.is_connected = False
        self._init_backend()

    def _init_backend(self):
        cfg = load_config()
        url = cfg.get("apps_script_url", "")
        if not url and "apps_script_url" in st.secrets:
            url = st.secrets["apps_script_url"]
        if url:
            self.apps_script_url = url.strip()
            self.is_connected = True

    def load_data(self):
        # 1. Try Apps Script Web App
        if self.apps_script_url:
            try:
                import requests
                res = requests.get(self.apps_script_url, timeout=5)
                if res.status_code == 200 and res.text.strip():
                    raw = res.json()
                    if isinstance(raw, dict) and "meals" in raw and "inventory" in raw:
                        # Cache to local file
                        with open(DATA_FILE, "w", encoding="utf-8") as f:
                            json.dump(raw, f, ensure_ascii=False, indent=2)
                        return raw
            except Exception as e:
                st.sidebar.warning(f"구글 시트 읽기 실패 (로컬 백업 사용): {e}")

        # 2. Local fallback
        if not os.path.exists(DATA_FILE):
            return {"inventory": [], "meals": [], "planner": [], "production_logs": []}
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_data(self, data):
        # Always save locally
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Sync to Google Apps Script Web App
        if self.apps_script_url:
            try:
                import requests
                requests.post(
                    self.apps_script_url,
                    data=json.dumps(data, ensure_ascii=False).encode('utf-8'),
                    headers={"Content-Type": "application/json"},
                    timeout=5
                )
            except Exception as e:
                st.sidebar.error(f"구글 시트 동기화 실패: {e}")

storage = StorageManager()
data = storage.load_data()

now = get_now()
current_year = now.year

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "172.30.1.50"

# Sidebar: Easy Google Sheets Setup & Status
with st.sidebar:
    st.header("☁️ 구글 시트 간편 연동 (대안 2)")
    cfg = load_config()
    cur_url = cfg.get("apps_script_url", "")

    if storage.apps_script_url:
        st.success("🟢 **Google Sheets 실시간 연동 중!**\n\n모든 큐브 변경/소진 내역이 내 구글 스프레드시트에 영구 저장됩니다.")
        c_ref, c_disc = st.columns(2)
        with c_ref:
            if st.button("🔄 시트 불러오기"):
                st.rerun()
        with c_disc:
            if st.button("🔌 연동 해제"):
                cfg["apps_script_url"] = ""
                save_config(cfg)
                storage.apps_script_url = ""
                st.rerun()
    else:
        st.info("🟡 **로컬 저장 모드**\n\n현재 로컬 파일에 저장 중입니다. 아래에서 구글 시트 웹앱 URL을 입력하면 영구 클라우드 저장이 시작됩니다.")

    with st.expander("⚡ 1분 만에 구글 시트 연동하기 (클릭)", expanded=not bool(cur_url)):
        st.markdown("""
        **순서대로 따라해 보세요:**
        1. [sheets.new](https://sheets.new) 접속 (새 시트 생성)
        2. 메뉴: **확장 프로그램** ➔ **Apps Script**
        3. 아래 코드 복사해 붙여넣고 **저장(Ctrl+S)**:
        ```javascript
        function doGet(e) {
          var ss = SpreadsheetApp.getActiveSpreadsheet();
          var sheet = ss.getSheetByName("db_state") || ss.getActiveSheet();
          var val = sheet.getRange("A1").getValue();
          return ContentService.createTextOutput(val || "{}").setMimeType(ContentService.MimeType.JSON);
        }
        function doPost(e) {
          var ss = SpreadsheetApp.getActiveSpreadsheet();
          var sheet = ss.getSheetByName("db_state") || ss.insertSheet("db_state");
          sheet.getRange("A1").setValue(e.postData.contents);
          return ContentService.createTextOutput(JSON.stringify({status: "ok"})).setMimeType(ContentService.MimeType.JSON);
        }
        ```
        4. 우측 상단 **[배포] ➔ [새 배포]**:
           * 유형: **웹 앱**
           * 액세스 권한: **모든 사용자(Anyone)** ⚠️
        5. 배포 후 나오는 **웹 앱 URL**을 아래에 붙여넣기!
        """)
        new_url = st.text_input("구글 웹앱 URL 붙여넣기", value=cur_url, placeholder="https://script.google.com/macros/s/.../exec")
        if st.button("🔗 연동 저장 & 즉시 동기화", type="primary"):
            if new_url.strip():
                cfg["apps_script_url"] = new_url.strip()
                save_config(cfg)
                storage.apps_script_url = new_url.strip()
                storage.save_data(data)
                st.success("🎉 구글 시트와 성공적으로 연결되었습니다!")
                st.rerun()
            else:
                st.warning("URL을 입력해주세요.")

    st.divider()
    st.caption(f"🕒 현재 기준 시각: **{now.strftime('%Y-%m-%d %H:%M')} (KST)**")

def parse_meal_date(date_str):
    m = re.search(r'(\d+)월\s*(\d+)일', date_str)
    if m:
        return datetime(current_year, int(m.group(1)), int(m.group(2))).date()
    return None

# Check if meal is eaten considering auto-deduction
def get_meal_eaten_status(meal, slot_type, auto_mode=True):
    override_key = f"{slot_type}_override"
    override = meal.get(override_key, None)
    
    if override == "skipped":
        return False, "❌ 건너뜀 (안 먹음 예외)"
    if override == "eaten":
        return True, "✅ 식사 완료 (수동 확인)"
    
    if meal.get(f"{slot_type}_eaten", False):
        return True, "✅ 식사 완료"

    if auto_mode:
        m_date = parse_meal_date(meal["date"])
        if m_date:
            if m_date < now.date():
                return True, "🤖 날짜 경과 (자동 차감 완료)"
            elif m_date == now.date():
                hour = now.hour
                if slot_type == "morning" and hour >= 10:
                    return True, "🤖 아침 시간 경과 (자동 차감 완료)"
                elif slot_type == "lunch" and hour >= 14:
                    return True, "🤖 점심 시간 경과 (자동 차감 완료)"
                elif slot_type == "dinner" and hour >= 19:
                    return True, "🤖 저녁 시간 경과 (자동 차감 완료)"

    return False, "⏳ 식사 대기 중"

def get_ingredient_category(name):
    for item in data.get("inventory", []):
        if item["name"] == name:
            return item.get("category", "채소류")
    return "채소류"

def get_badge_class(cat):
    if cat == "곡류": return "badge-grain"
    if cat == "단백질": return "badge-protein"
    return "badge-veg"

# Calculate Chronological Depletion & Deadlines with Automatic Deduction
def calculate_system_state(data, auto_mode=True):
    meals = data.get("meals", [])
    inventory = data.get("inventory", [])
    prod_logs = data.get("production_logs", [])

    consumed_counts = {}
    uneaten_slots = []

    for m_idx, m in enumerate(meals):
        d = m["date"]
        dow = m["day_of_week"]
        for slot_type, t_name in [("morning", "아침"), ("lunch", "점심"), ("dinner", "저녁")]:
            is_eaten, reason = get_meal_eaten_status(m, slot_type, auto_mode)
            ingredients = m.get(slot_type, [])
            if is_eaten:
                for ing in ingredients:
                    consumed_counts[ing] = consumed_counts.get(ing, 0) + 1
            else:
                uneaten_slots.append({
                    "slot_idx": len(uneaten_slots),
                    "meal_idx": m_idx,
                    "date": d,
                    "dow": dow,
                    "time": t_name,
                    "label": f"{d}({dow}) {t_name}",
                    "ingredients": ingredients
                })

    produced_counts = {}
    for log in prod_logs:
        ing = log.get("ingredient")
        qty = log.get("quantity", 0)
        produced_counts[ing] = produced_counts.get(ing, 0) + qty

    timeline_results = {}
    shortages = []

    for item in inventory:
        name = item["name"]
        init = item.get("initial_stock", item.get("current_stock", 0))
        produced = produced_counts.get(name, 0)
        consumed = consumed_counts.get(name, 0)
        manual_adj = item.get("manual_adjustment", 0)

        calc_stock = max(0, init + produced + manual_adj - consumed)
        item["current_stock"] = calc_stock

        rem = calc_stock
        depletion_slot = None
        first_deficit_slot = None
        future_need = 0

        for s in uneaten_slots:
            if name in s["ingredients"]:
                future_need += 1
                if rem > 0:
                    rem -= 1
                    if rem == 0:
                        depletion_slot = s
                else:
                    rem -= 1
                    if first_deficit_slot is None:
                        first_deficit_slot = s

        if calc_stock == 0:
            if future_need > 0:
                depletion_desc = "🚨 현재 0개 (소진 상태)"
            else:
                depletion_desc = "재고 없음 (이후 식단 계획 없음)"
        elif depletion_slot:
            depletion_desc = f"📅 {depletion_slot['label']} 식사 시 마지막 큐브 소진"
        else:
            if calc_stock >= future_need and future_need > 0:
                depletion_desc = f"✅ 15일 계획 내 소진 안 됨 (잔여 {calc_stock - future_need}개 여유)"
            elif future_need == 0:
                depletion_desc = f"✅ 남은 식단에 사용 없음 (보유 {calc_stock}개)"
            else:
                depletion_desc = "여유"

        urgency_score = 9999
        deadline_desc = None
        deadline_dday = None

        if future_need > calc_stock:
            diff = future_need - calc_stock
            if first_deficit_slot:
                urgency_score = first_deficit_slot["slot_idx"]
                d_date = first_deficit_slot["date"]
                d_dow = first_deficit_slot["dow"]
                d_time = first_deficit_slot["time"]

                if d_time == "아침":
                    deadline_desc = f"🚨 {d_date}({d_dow}) '전날 밤'까지 무조건 제작 필수!"
                elif d_time == "점심":
                    deadline_desc = f"⚠️ {d_date}({d_dow}) '당일 오전 10시' (또는 전날 밤)까지 제작 필수!"
                else:
                    deadline_desc = f"⚠️ {d_date}({d_dow}) '당일 오후 3시' 전까지 제작 필수!"

                slot_dist = first_deficit_slot["slot_idx"]
                if slot_dist <= 1:
                    deadline_dday = "D-0 (당장 오늘!)"
                elif slot_dist <= 4:
                    deadline_dday = "D-1 (내일 마감)"
                elif slot_dist <= 7:
                    deadline_dday = "D-2 (모레 마감)"
                else:
                    day_dist = slot_dist // 3
                    deadline_dday = f"D-{day_dist}"
            else:
                deadline_desc = "식단 일정 내 제작 필요"
                deadline_dday = "여유"

            shortages.append({
                "name": name,
                "category": item["category"],
                "stock": calc_stock,
                "total_need": future_need,
                "shortage": diff,
                "depletion_desc": depletion_desc,
                "first_deficit_slot": first_deficit_slot["label"] if first_deficit_slot else "식단 후반부",
                "deadline_desc": deadline_desc,
                "deadline_dday": deadline_dday,
                "urgency_score": urgency_score
            })

        timeline_results[name] = {
            "stock": calc_stock,
            "future_need": future_need,
            "net": calc_stock - future_need,
            "depletion_desc": depletion_desc,
            "depletion_slot": depletion_slot
        }

    shortages.sort(key=lambda x: (x["urgency_score"], -x["shortage"]))

    # Calculate surplus items (items with stock > future_need)
    surplus_items = []
    for item in inventory:
        name = item["name"]
        t_info = timeline_results.get(name, {})
        net = t_info.get("net", 0)
        if net > 0:
            surplus_items.append({
                "name": name,
                "category": item["category"],
                "stock": t_info.get("stock", 0),
                "future_need": t_info.get("future_need", 0),
                "surplus": net
            })
    surplus_items.sort(key=lambda x: -x["surplus"])

    return timeline_results, shortages, consumed_counts, surplus_items

# Header
local_ip = get_local_ip()
col_title, col_ip = st.columns([2, 1])
with col_title:
    st.title("🥣 우리 아기 이유식 & 큐브 플래너")
with col_ip:
    st.markdown(f"""
    <div class="mobile-ip-box">
        📱 <b>스마트폰으로 접속하기</b><br>
        로컬 Wi-Fi 주소:<br>
        <code style="color:#ffffff; background:#4338ca; padding:2px 6px; border-radius:4px;">http://{local_ip}:8501</code>
    </div>
    """, unsafe_allow_html=True)

if "auto_deduct" not in st.session_state:
    st.session_state.auto_deduct = True

timeline_results, urgent_shortages, consumed_counts, surplus_items = calculate_system_state(data, auto_mode=st.session_state.auto_deduct)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🍽️ 오늘의 식단 & 자동 차감",
    "🧊 큐브 제작 (입고)",
    "📊 냉동실 큐브 재고 (소진일)",
    "🛒 마트 장보기 (마감일 순)",
    "📅 15일 식단표 전체보기"
])

# ==========================================
# TAB 1: 오늘의 식단 & 스마트 자동 차감
# ==========================================
with tab1:
    st.markdown(f"""
    <div class="auto-mode-banner">
        🤖 <b>스마트 자동 차감 기능 작동 중</b><br>
        매일 날짜와 끼니 시간(아침 10시, 점심 14시, 저녁 19시)이 지나면 <b>버튼을 누르지 않아도 냉동실 재고가 알아서 자동 차감</b>됩니다.<br>
        <small>※ 아기가 안 먹었거나 건너뛴 끼니만 <b>'❌ 건너뜀 (안 먹음)'</b> 버튼을 눌러 재고를 보존해 주세요.</small>
    </div>
    """, unsafe_allow_html=True)

    c_date_sel, c_toggle = st.columns([3, 1])
    meals = data.get("meals", [])
    if not meals:
        st.warning("식단표 데이터가 없습니다.")
    else:
        date_options = [f"{m['date']} ({m['day_of_week']})" for m in meals]
        
        today_str = f"{now.month}월{now.day}일"
        default_idx = 0
        for idx, m in enumerate(meals):
            if m["date"].replace(" ", "") == today_str:
                default_idx = idx
                break

        with c_date_sel:
            selected_date_str = st.selectbox("📅 조회/관리 날짜 선택", date_options, index=default_idx)
            selected_idx = date_options.index(selected_date_str)
            meal_entry = meals[selected_idx]

        with c_toggle:
            st.session_state.auto_deduct = st.toggle("🤖 자동 차감 모드", value=True)

        if meal_entry.get("note"):
            st.info(f"💡 **특이사항/소진 예정**: {meal_entry['note']}")

        col_m, col_l, col_d = st.columns(3)

        def set_override(meal_obj, slot_type, override_val):
            meal_obj[f"{slot_type}_override"] = override_val
            storage.save_data(data)
            st.rerun()

        # Morning
        with col_m:
            st.markdown("### 🌅 아침 식단")
            ing_html = "".join([f'<span class="tag-badge {get_badge_class(get_ingredient_category(x))}">{x}</span>' for x in meal_entry.get("morning", [])])
            st.markdown(ing_html, unsafe_allow_html=True)
            st.write("")
            is_m_eaten, m_status_text = get_meal_eaten_status(meal_entry, "morning", st.session_state.auto_deduct)
            if is_m_eaten:
                st.success(m_status_text)
                if st.button("❌ 아침 안 먹음 (건너뜀/재고복구)", key="skip_m"):
                    set_override(meal_entry, "morning", "skipped")
            else:
                st.info(m_status_text)
                if st.button("🥣 지금 바로 식사 완료 처리", key="force_eat_m", type="primary", use_container_width=True):
                    set_override(meal_entry, "morning", "eaten")

            with st.popover("✏️ 아침 재료 관리/삭제"):
                st.caption(f"🌅 {selected_date_str} 아침 재료")
                for ing in list(meal_entry.get("morning", [])):
                    c_txt, c_del = st.columns([3, 1])
                    c_txt.write(f"• {ing}")
                    if c_del.button("삭제", key=f"del_m_{selected_idx}_{ing}"):
                        meal_entry["morning"].remove(ing)
                        storage.save_data(data)
                        st.rerun()

        # Lunch
        with col_l:
            st.markdown("### ☀️ 점심 식단")
            ing_html = "".join([f'<span class="tag-badge {get_badge_class(get_ingredient_category(x))}">{x}</span>' for x in meal_entry.get("lunch", [])])
            st.markdown(ing_html, unsafe_allow_html=True)
            st.write("")
            is_l_eaten, l_status_text = get_meal_eaten_status(meal_entry, "lunch", st.session_state.auto_deduct)
            if is_l_eaten:
                st.success(l_status_text)
                if st.button("❌ 점심 안 먹음 (건너뜀/재고복구)", key="skip_l"):
                    set_override(meal_entry, "lunch", "skipped")
            else:
                st.info(l_status_text)
                if st.button("🥣 지금 바로 식사 완료 처리", key="force_eat_l", type="primary", use_container_width=True):
                    set_override(meal_entry, "lunch", "eaten")

            with st.popover("✏️ 점심 재료 관리/삭제"):
                st.caption(f"☀️ {selected_date_str} 점심 재료")
                for ing in list(meal_entry.get("lunch", [])):
                    c_txt, c_del = st.columns([3, 1])
                    c_txt.write(f"• {ing}")
                    if c_del.button("삭제", key=f"del_l_{selected_idx}_{ing}"):
                        meal_entry["lunch"].remove(ing)
                        storage.save_data(data)
                        st.rerun()

        # Dinner
        with col_d:
            st.markdown("### 🌙 저녁 식단")
            ing_html = "".join([f'<span class="tag-badge {get_badge_class(get_ingredient_category(x))}">{x}</span>' for x in meal_entry.get("dinner", [])])
            st.markdown(ing_html, unsafe_allow_html=True)
            st.write("")
            is_d_eaten, d_status_text = get_meal_eaten_status(meal_entry, "dinner", st.session_state.auto_deduct)
            if is_d_eaten:
                st.success(d_status_text)
                if st.button("❌ 저녁 안 먹음 (건너뜀/재고복구)", key="skip_d"):
                    set_override(meal_entry, "dinner", "skipped")
            else:
                st.info(d_status_text)
                if st.button("🥣 지금 바로 식사 완료 처리", key="force_eat_d", type="primary", use_container_width=True):
                    set_override(meal_entry, "dinner", "eaten")

            with st.popover("✏️ 저녁 재료 관리/삭제"):
                st.caption(f"🌙 {selected_date_str} 저녁 재료")
                for ing in list(meal_entry.get("dinner", [])):
                    c_txt, c_del = st.columns([3, 1])
                    c_txt.write(f"• {ing}")
                    if c_del.button("삭제", key=f"del_d_{selected_idx}_{ing}"):
                        meal_entry["dinner"].remove(ing)
                        storage.save_data(data)
                        st.rerun()

        # ==========================================
        # 💡 [냉동실 털기] 빨리 소진해야 하는 큐브 추천 & 식단에 추가
        # ==========================================
        st.write("")
        st.divider()
        st.subheader("💡 [냉동실 털기] 빨리 소진해야 하는 큐브 추천 & 식단에 추가")
        st.caption("남은 15일 식단에 계획된 수량보다 냉동실에 더 많이 남아있는 **잉여 큐브들을 추천**해 드립니다. 선택한 끼니에 추가하면 빠르게 소진할 수 있습니다!")

        if not surplus_items:
            st.success("🎉 현재 냉동실에 남아도는 잉여 큐브가 없습니다. 식단표와 완벽히 맞아떨어집니다!")
        else:
            surplus_html = " &nbsp;&nbsp;|&nbsp;&nbsp; ".join([
                f"<b>{s['name']}</b> ({s['category']}): <b style='color:#ef4444;'>+{s['surplus']}개 남아돌음</b> (냉동실 {s['stock']}개 / 남은필요 {s['future_need']}개)"
                for s in surplus_items
            ])
            st.markdown(f"""
            <div style="background-color:#fff7ed; border:1px solid #fed7aa; border-radius:8px; padding:12px; margin-bottom:12px;">
                📢 <b>우선 소진 권장 품목 (잉여 큐브):</b><br>
                {surplus_html}
            </div>
            """, unsafe_allow_html=True)

            c_add_slot, c_add_item, c_add_btn = st.columns([2, 2, 2])
            with c_add_slot:
                target_slot = st.selectbox("추가할 끼니 선택", ["🌅 아침", "☀️ 점심", "🌙 저녁"], key="surplus_slot_choice")
            with c_add_item:
                surplus_options = [f"{s['name']} (+{s['surplus']}개 남음)" for s in surplus_items]
                chosen_opt = st.selectbox("추가할 잉여 큐브", surplus_options, key="surplus_item_choice")
                chosen_ing_name = chosen_opt.split(" (")[0]
            with c_add_btn:
                st.write("")
                st.write("")
                if st.button("➕ 선택한 끼니에 큐브 추가하기", type="primary", use_container_width=True, key="btn_add_surplus"):
                    slot_map = {"🌅 아침": "morning", "☀️ 점심": "lunch", "🌙 저녁": "dinner"}
                    slot_key = slot_map[target_slot]
                    if chosen_ing_name in meal_entry.get(slot_key, []):
                        st.warning(f"이미 {selected_date_str} {target_slot} 식단에 '{chosen_ing_name}'이(가) 포함되어 있습니다!")
                    else:
                        if slot_key not in meal_entry:
                            meal_entry[slot_key] = []
                        meal_entry[slot_key].append(chosen_ing_name)
                        storage.save_data(data)
                        st.success(f"🎉 {selected_date_str} {target_slot}에 '{chosen_ing_name}' 큐브가 추가되었습니다! 냉동실 잉여 재고가 소진됩니다.")
                        st.rerun()

# ==========================================
# TAB 2: 큐브 제작 & 입고 등록
# ==========================================
with tab2:
    st.subheader("🧊 큐브 새로 만들었을 때 바로 입고 등록")
    st.caption("새로 만든 큐브를 등록하면 냉동실 재고, 예상 소진일, 마감일 알림이 구글 시트와 클라우드에 즉시 저장됩니다.")

    inv_names = [item["name"] for item in data.get("inventory", [])]
    inv_names.sort()

    with st.form("make_cube_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            sel_mode = st.radio("재료 선택 방식", ["기존 재료 선택", "➕ 새 재료 직접 추가"], horizontal=True)
            if sel_mode == "기존 재료 선택":
                cube_ingredient = st.selectbox("품목 (재료)", inv_names)
                cube_category = get_ingredient_category(cube_ingredient)
            else:
                cube_ingredient = st.text_input("새 재료 이름 (예: 콜리플라워, 비트)")
                cube_category = st.selectbox("분류", ["채소류", "단백질", "곡류", "과일/기타"])

        with c2:
            cube_qty = st.number_input("제작 수량 (개)", min_value=1, max_value=60, value=12, step=1)
            today_date = now.date()
            make_date = st.date_input("제작일", value=today_date)
            exp_date = st.date_input("권장 유통기한 (제작일 + 3주)", value=make_date + timedelta(days=21))

        memo = st.text_input("메모 (선택사항)", placeholder="예: 15g 큐브 12구 1판, 무항생제 닭안심 사용")

        submit_cube = st.form_submit_button("🧊 냉동실 큐브 입고 등록 (+ 반영하기)", type="primary", use_container_width=True)

        if submit_cube:
            if not cube_ingredient.strip():
                st.error("재료명을 입력해주세요.")
            else:
                ing_name = cube_ingredient.strip()
                found = False
                for item in data["inventory"]:
                    if item["name"] == ing_name:
                        found = True
                        break
                if not found:
                    data["inventory"].append({
                        "category": cube_category,
                        "name": ing_name,
                        "initial_stock": 0,
                        "current_stock": 0,
                        "manual_adjustment": 0
                    })

                if "production_logs" not in data:
                    data["production_logs"] = []
                data["production_logs"].insert(0, {
                    "date": str(make_date),
                    "ingredient": ing_name,
                    "category": cube_category,
                    "quantity": int(cube_qty),
                    "exp_date": str(exp_date),
                    "memo": memo
                })

                storage.save_data(data)
                st.success(f"🎉 [{ing_name}] 큐브 +{cube_qty}개가 성공적으로 입고되었습니다!")
                st.rerun()

    st.divider()
    st.subheader("📋 최근 큐브 제작 (입고) 내역")
    logs = data.get("production_logs", [])
    if not logs:
        st.info("아직 등록된 큐브 제작 내역이 없습니다. 위에서 제작한 큐브를 등록해 보세요!")
    else:
        for idx, log in enumerate(logs):
            c_info, c_del = st.columns([5, 1])
            with c_info:
                st.markdown(f"""
                **{log['date']}** | **{log['ingredient']}** `+{log['quantity']}개` ({log['category']})  
                <small>유통기한: {log.get('exp_date', '-')} | 메모: {log.get('memo', '-')}</small>
                """, unsafe_allow_html=True)
            with c_del:
                if st.button("삭제", key=f"del_log_{idx}"):
                    data["production_logs"].pop(idx)
                    storage.save_data(data)
                    st.rerun()
            st.divider()

# ==========================================
# TAB 3: 냉동실 큐브 재고 현황 (소진일)
# ==========================================
with tab3:
    st.subheader("📊 냉동실 큐브 실시간 재고 & 예상 소진 날짜")
    st.caption("날짜 경과에 따라 자동 차감된 **현재 실재고**와 **마지막 1개가 떨어지는 날짜**를 정확히 알려드립니다.")

    inv = data.get("inventory", [])
    total_cubes = sum(timeline_results.get(item["name"], {}).get("stock", 0) for item in inv)
    urgent_count = sum(1 for item in inv if timeline_results.get(item["name"], {}).get("net", 0) < 0)
    safe_count = len(inv) - urgent_count

    m1, m2, m3 = st.columns(3)
    m1.metric("🧊 총 보관 큐브 수", f"{total_cubes} 개")
    m2.metric("🚨 향후 부족/제작 필요", f"{urgent_count} 종", delta="장보기 대상", delta_color="inverse")
    m3.metric("✅ 수량 여유/충분", f"{safe_count} 종")

    if surplus_items:
        surplus_summary = ", ".join([f"**{s['name']}** (+{s['surplus']}개 잉여)" for s in surplus_items])
        st.markdown(f"""
        <div style="background-color:#fff7ed; border-left:5px solid #f97316; border-radius:8px; padding:10px 14px; margin-top:8px; margin-bottom:12px; font-size:0.95rem; color:#9a3412;">
            💡 <b>냉동실 털기 추천:</b> 남은 식단보다 재고가 많은 큐브: {surplus_summary}<br>
            <small>➔ <b>[🍽️ 오늘의 식단 & 자동 차감]</b> 탭에서 원하는 끼니에 큐브를 추가하여 빠르게 소진할 수 있습니다!</small>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    filter_cat = st.radio("카테고리 필터", ["전체", "단백질", "채소류", "곡류"], horizontal=True)

    filtered_inv = [item for item in inv if filter_cat == "전체" or item["category"] == filter_cat]

    for item in filtered_inv:
        name = item["name"]
        t_info = timeline_results.get(name, {})
        stock = t_info.get("stock", 0)
        need = t_info.get("future_need", 0)
        net = t_info.get("net", 0)
        dep_desc = t_info.get("depletion_desc", "-")

        if stock <= 0 and need > 0:
            card_class = "card-urgent"
            badge_html = '<span style="color:#ef4444; font-weight:bold;">🚨 소진됨 (부족)</span>'
        elif net < 0:
            card_class = "card-warning"
            badge_html = '<span style="color:#d97706; font-weight:bold;">⚠️ 곧 소진됨 (제작 필요)</span>'
        else:
            card_class = "card-good"
            badge_html = '<span style="color:#10b981; font-weight:bold;">✅ 충분/여유</span>'

        c_card, c_adj = st.columns([4, 2])
        with c_card:
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                    <div>
                        <b style="font-size:1.1rem;">{name}</b> 
                        <span class="tag-badge {get_badge_class(item['category'])}">{item['category']}</span>
                    </div>
                    <div>{badge_html}</div>
                </div>
                <div style="margin-top:4px; font-size:0.95rem;">
                    🧊 <b>현재 실재고:</b> <span style="font-size:1.15rem; color:#2563eb; font-weight:bold;">{stock}개</span> &nbsp;|&nbsp; 
                    📋 <b>남은 식단 필요량:</b> {need}개 &nbsp;(잔여 예측: <b>{'+' if net>0 else ''}{net}개</b>)
                </div>
                <div style="margin-top:6px; padding:6px 10px; background:rgba(255,255,255,0.7); border-radius:6px; font-size:0.9rem;">
                    ⏰ <b>예상 소진 시점:</b> <b>{dep_desc}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c_adj:
            st.write("")
            b_minus, b_plus = st.columns(2)
            with b_minus:
                if st.button("➖ 1", key=f"dec_{name}", use_container_width=True):
                    item["manual_adjustment"] = item.get("manual_adjustment", 0) - 1
                    storage.save_data(data)
                    st.rerun()
            with b_plus:
                if st.button("➕ 1", key=f"inc_{name}", use_container_width=True):
                    item["manual_adjustment"] = item.get("manual_adjustment", 0) + 1
                    storage.save_data(data)
                    st.rerun()

# ==========================================
# TAB 4: 마트 장보기 추천 & 무조건 제작 마감일 플래너
# ==========================================
with tab4:
    st.subheader("🛒 마트 장보기 추천 목록 (급한 순서대로 정렬)")
    st.caption("날짜 경과에 따라 소진된 현재 재고를 바탕으로 **가장 먼저 부족해지는 순서대로 정렬**하고 **무조건 완성해야 하는 최종 마감일**을 알려드립니다.")

    if not urgent_shortages:
        st.success("🎉 현재 계획된 남은 식단에 부족한 큐브가 없습니다! 모든 재료가 넉넉합니다.")
    else:
        st.error(f"총 {len(urgent_shortages)}개 품목이 부족합니다. 아래 마감일을 확인하여 장보기 및 큐브 제작을 진행해 주세요.")

        for rank, s in enumerate(urgent_shortages, 1):
            dday_val = s["deadline_dday"]
            if "D-0" in dday_val:
                badge_class = "badge-dday-urgent"
                border_class = "shop-item-card"
            elif "D-1" in dday_val or "D-2" in dday_val:
                badge_class = "badge-dday-warn"
                border_class = "shop-item-card-warn"
            else:
                badge_class = "badge-dday-safe"
                border_class = "shop-item-card"

            st.markdown(f"""
            <div class="{border_class}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-size:1.15rem; font-weight:bold;">
                        #{rank} {s['name']} <span class="tag-badge {get_badge_class(s['category'])}">{s['category']}</span>
                    </div>
                    <div>
                        <span class="{badge_class}">{s['deadline_dday']}</span>
                    </div>
                </div>
                <div style="margin-top:8px; font-size:1.02rem; color:#b91c1c; font-weight:bold;">
                    🎯 제작 마감일: {s['deadline_desc']}
                </div>
                <div style="margin-top:6px; font-size:0.92rem; color:#4b5563;">
                    • <b>부족 수량:</b> <b style="color:#ef4444;">+{s['shortage']}개 부족</b> (현재 보유 {s['stock']}개 / 남은 필요 {s['total_need']}개)<br>
                    • <b>첫 부족 발생 끼니:</b> <span style="background:#fee2e2; padding:1px 6px; border-radius:4px; font-weight:bold;">{s['first_deficit_slot']}부터 없음!</span><br>
                    • <b>현재 재고 소진일:</b> {s['depletion_desc']}
                </div>
                <div style="margin-top:8px;">
                    🛒 <b>장보기 메모:</b> <code>{s['name']}</code> 신선 재료 구매 후 12구 틀 기준 {max(1, (s['shortage'] + 11) // 12)}판 제작 권장
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.subheader("🗓️ 권장 큐브 데이 4단계 로드맵")
    planner = data.get("planner", [])
    for p in planner:
        with st.expander(f"📌 {p['stage']} : {p['d_day']} ({p['items']})", expanded=True):
            st.write(f"- **소진 직전 품목**: {p['items']}")
            st.write(f"- **최소 부족 수량**: {p['shortage']}")
            st.write(f"- **권장 제작 규격**: {p['tray']}")
            st.write(f"- **장보기 추천 재료**: `{p['shopping']}`")

# ==========================================
# TAB 5: 15일 식단표 전체보기
# ==========================================
with tab5:
    st.subheader("📅 전체 15일 이유식 식단표")
    
    table_rows = []
    for m in data.get("meals", []):
        m_eaten, _ = get_meal_eaten_status(m, "morning", st.session_state.auto_deduct)
        l_eaten, _ = get_meal_eaten_status(m, "lunch", st.session_state.auto_deduct)
        d_eaten, _ = get_meal_eaten_status(m, "dinner", st.session_state.auto_deduct)
        table_rows.append({
            "날짜": m["date"],
            "요일": m["day_of_week"],
            "아침": f"{'✅' if m_eaten else '⬜'} " + ", ".join(m.get("morning", [])),
            "점심": f"{'✅' if l_eaten else '⬜'} " + ", ".join(m.get("lunch", [])),
            "저녁": f"{'✅' if d_eaten else '⬜'} " + ", ".join(m.get("dinner", [])),
            "비고": m.get("note", "")
        })

    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    st.divider()
    if st.button("📥 현재 상태 엑셀(XLSX)로 내보내기/저장"):
        try:
            import openpyxl
            wb = openpyxl.Workbook()
            ws1 = wb.active
            ws1.title = "현재재고_및_소진일"
            ws1.append(["분류", "품목", "현재실재고", "남은필요량", "예상잔여", "예상소진시점"])
            for item in data.get("inventory", []):
                n = item["name"]
                t_info = timeline_results.get(n, {})
                s = t_info.get("stock", 0)
                nd = t_info.get("future_need", 0)
                dep = t_info.get("depletion_desc", "-")
                ws1.append([item["category"], n, s, nd, s - nd, dep])
            
            ws2 = wb.create_sheet(title="식단표")
            ws2.append(["날짜", "요일", "아침", "점심", "저녁", "아침완료", "점심완료", "저녁완료", "비고"])
            for m in data.get("meals", []):
                m_eaten, _ = get_meal_eaten_status(m, "morning", st.session_state.auto_deduct)
                l_eaten, _ = get_meal_eaten_status(m, "lunch", st.session_state.auto_deduct)
                d_eaten, _ = get_meal_eaten_status(m, "dinner", st.session_state.auto_deduct)
                ws2.append([
                    m["date"], m["day_of_week"],
                    "\n".join(m.get("morning", [])),
                    "\n".join(m.get("lunch", [])),
                    "\n".join(m.get("dinner", [])),
                    "완료" if m_eaten else "미완료",
                    "완료" if l_eaten else "미완료",
                    "완료" if d_eaten else "미완료",
                    m.get("note", "")
                ])
            wb.save("이유식_식단_및_큐브관리_최신현황.xlsx")
            st.success("✅ '이유식_식단_및_큐브관리_최신현황.xlsx' 파일로 저장 완료되었습니다!")
        except Exception as e:
            st.error(f"엑셀 저장 오류: {e}")
