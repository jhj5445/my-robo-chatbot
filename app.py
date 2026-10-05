import streamlit as st
import json
import os
import socket
import re
from datetime import datetime, timedelta, timezone, date

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
        padding: 8px 12px;
        margin-bottom: 12px;
        font-size: 0.88rem;
        color: #1e40af;
    }

    /* 📱 Mobile Optimized Media Queries */
    @media (max-width: 768px) {
        .block-container {
            padding: 0.5rem 0.6rem 2.5rem 0.6rem !important;
        }
        .mobile-ip-box {
            display: none !important;
        }
        .auto-mode-banner {
            display: none !important;
        }
        .stButton button {
            min-height: 38px !important;
            font-size: 0.86rem !important;
            border-radius: 8px !important;
            padding: 4px 10px !important;
        }
        .tag-badge {
            font-size: 0.78rem !important;
            padding: 2px 6px !important;
            margin-right: 3px !important;
            margin-bottom: 3px !important;
        }
        .card-urgent, .card-warning, .card-good {
            padding: 8px 10px !important;
            margin-bottom: 6px !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px !important;
            overflow-x: auto !important;
            flex-wrap: nowrap !important;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 5px 8px !important;
            font-size: 0.8rem !important;
            white-space: nowrap !important;
        }
        div[role="radiogroup"] {
            background: #f1f5f9 !important;
            padding: 3px !important;
            border-radius: 10px !important;
            gap: 4px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Hybrid Storage Manager (GitHub API & Google Sheets & Local)
# ==========================================
import base64
import requests

DEFAULT_APPS_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbxUVJye2HTxhCHH08VnrhhzK9e7a3pfpK7-nGYQQX9wAqFIkxJbCidNUiDef7lsbabxhg/exec"
CONFIG_FILE = "config.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"apps_script_url": DEFAULT_APPS_SCRIPT_URL}

def save_config(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

class GitHubBackend:
    def __init__(self, repo, token, path="data.json"):
        self.repo = repo.strip()
        self.token = token.strip()
        self.path = path
        self.base_url = f"https://api.github.com/repos/{self.repo}/contents/{self.path}"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }
        self.cached_sha = None

    def load(self):
        try:
            res = requests.get(self.base_url, headers=self.headers, timeout=6)
            if res.status_code == 200:
                info = res.json()
                self.cached_sha = info.get("sha")
                content_str = base64.b64decode(info["content"]).decode("utf-8")
                return json.loads(content_str)
        except Exception:
            pass
        return None

    def save(self, data):
        try:
            content_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
            b64_content = base64.b64encode(content_bytes).decode("utf-8")
            for attempt in range(2):
                sha = self.cached_sha
                if not sha or attempt > 0:
                    res_get = requests.get(self.base_url, headers=self.headers, timeout=5)
                    if res_get.status_code == 200:
                        sha = res_get.json().get("sha")
                        self.cached_sha = sha

                now_str = get_now().strftime("%Y-%m-%d %H:%M:%S")
                payload = {
                    "message": f"👶 이유식 큐브 데이터 업데이트 ({now_str})",
                    "content": b64_content
                }
                if sha:
                    payload["sha"] = sha

                res = requests.put(self.base_url, json=payload, headers=self.headers, timeout=8)
                if res.status_code in [200, 201]:
                    self.cached_sha = res.json().get("content", {}).get("sha")
                    return True, "성공"
                elif res.status_code == 409:
                    self.cached_sha = None
                    continue
                elif res.status_code == 403:
                    return False, "권한 오류 (403): GitHub 토큰에 쓰기 권한이 없습니다. Fine-grained 토큰 설정의 'Repository permissions' -> 'Contents'를 [Read and write]로 변경하시거나, Classic 토큰(ghp_)으로 'repo'를 체크하여 발급해주세요."
                else:
                    return False, f"HTTP {res.status_code}: {res.text[:100]}"
            return False, "동시성 충돌"
        except Exception as e:
            return False, str(e)

class StorageManager:
    def __init__(self):
        self.github = None
        self.apps_script_url = None
        self.mode = "local"
        self._init_backend()

    def _init_backend(self):
        cfg = load_config()
        
        # 1. GitHub API Priority
        gh_repo = cfg.get("github_repo", "")
        gh_token = cfg.get("github_token", "")
        if not gh_repo or not gh_token:
            try:
                gh_repo = gh_repo or st.secrets.get("github_repo", "")
                gh_token = gh_token or st.secrets.get("github_token", "")
            except Exception:
                pass

        if gh_repo and gh_token:
            self.github = GitHubBackend(gh_repo, gh_token)
            self.mode = "github"
            return

        # 2. Google Apps Script Web App
        url = cfg.get("apps_script_url", "") or st.secrets.get("apps_script_url", DEFAULT_APPS_SCRIPT_URL)
        if url:
            self.apps_script_url = url.strip()
            self.mode = "apps_script"

    def load_data(self):
        if self.mode == "github" and self.github:
            gh_data = self.github.load()
            if gh_data:
                with open(DATA_FILE, "w", encoding="utf-8") as f:
                    json.dump(gh_data, f, ensure_ascii=False, indent=2)
                return gh_data

        if self.mode == "apps_script" and self.apps_script_url:
            try:
                res = requests.get(self.apps_script_url, timeout=5)
                if res.status_code == 200 and res.text.strip():
                    raw = res.json()
                    if isinstance(raw, dict) and "meals" in raw and "inventory" in raw:
                        with open(DATA_FILE, "w", encoding="utf-8") as f:
                            json.dump(raw, f, ensure_ascii=False, indent=2)
                        return raw
            except Exception:
                pass

        if not os.path.exists(DATA_FILE):
            return {"inventory": [], "meals": [], "planner": [], "production_logs": []}
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_data(self, data):
        # Update session cache
        if "app_data" in st.session_state:
            st.session_state.app_data = data

        # Always cache locally
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        if self.mode == "github" and self.github:
            ok, msg = self.github.save(data)
            return ok, msg

        if self.mode == "apps_script" and self.apps_script_url:
            try:
                requests.post(
                    self.apps_script_url,
                    data=json.dumps(data, ensure_ascii=False).encode('utf-8'),
                    headers={"Content-Type": "application/json"},
                    timeout=5
                )
                return True, "성공"
            except Exception as e:
                return False, str(e)

        return True, "로컬 저장"

storage = StorageManager()

# Session State Cache for fast responsive navigation
if "app_data" not in st.session_state:
    st.session_state.app_data = storage.load_data()

data = st.session_state.app_data

now = get_now()
current_year = now.year

# Sidebar: Storage Sync Settings
with st.sidebar:
    st.header("☁️ 클라우드 데이터베이스 연동")
    cfg = load_config()

    if storage.mode == "github" and storage.github:
        st.success(f"🟢 **GitHub 저장소 실시간 연동 중!**\n\n저장소: `{storage.github.repo}`\n\n모든 큐브 변경/소진 내역이 `data.json`에 **자동 커밋**됩니다.")
        c_save, c_ref = st.columns(2)
        with c_save:
            if st.button("🐙 GitHub에 저장", type="primary", use_container_width=True, key="side_save_btn"):
                with st.spinner("GitHub 커밋 푸시 중..."):
                    ok, msg = storage.save_data(data)
                if ok:
                    st.toast("✅ GitHub에 커밋 완료!")
                    st.success(f"✅ 커밋 완료! ({get_now().strftime('%H:%M:%S')})")
                else:
                    st.error(f"❌ 실패: {msg}")
        with c_ref:
            if st.button("🔄 최신 불러오기", use_container_width=True, key="side_reload_btn"):
                st.session_state.app_data = storage.load_data()
                st.rerun()

        if st.button("🔌 GitHub 연동 해제", key="side_disc_gh"):
            cfg["github_token"] = ""
            cfg["github_repo"] = ""
            save_config(cfg)
            storage.mode = "local"
            storage.github = None
            st.rerun()

    elif storage.mode == "apps_script":
        st.info("🟢 **구글 시트 연동 모드 작동 중**")
        c_save, c_ref = st.columns(2)
        with c_save:
            if st.button("☁️ 시트에 저장", type="primary", use_container_width=True, key="side_save_gs"):
                storage.save_data(data)
                st.toast("✅ 구글 시트에 저장 완료!")
                st.success(f"✅ 저장 완료 ({get_now().strftime('%H:%M:%S')})")
        with c_ref:
            if st.button("🔄 시트 불러오기", use_container_width=True, key="side_reload_gs"):
                st.rerun()

    # Expandable GitHub Connection Setup
    with st.expander("🐙 GitHub 저장소 연동 설정 (추천 ⭐)", expanded=(storage.mode != "github")):
        st.markdown("""
        **GitHub 연동 방법:**
        - **방법 1 (가장 간단 - Classic 토큰 추천 ⭐):**  
          1. [github.com/settings/tokens](https://github.com/settings/tokens) 접속  
          2. **Generate new token (classic)** 클릭  
          3. **`repo`** (최상단 체크박스) 체크 후 하단 발급 ➔ 토큰(`ghp_...`) 복사
        - **방법 2 (현재 Fine-grained 토큰 수정 시):**  
          - 기존 토큰 페이지에서 **Repository permissions** ➔ **`Contents`**를 **[Read and write]**로 변경하고 저장하시면 즉시 정상 작동합니다!
        """)
        in_repo = st.text_input("GitHub 저장소 (아이디/저장소명)", value=cfg.get("github_repo", ""), placeholder="예: username/baby-food")
        in_token = st.text_input("GitHub 토큰 (ghp_... 또는 github_pat_...)", value=cfg.get("github_token", ""), type="password", placeholder="ghp_xxxxxxxxxxxx")
        if st.button("🔗 GitHub 연동하기", type="primary", key="btn_connect_github"):
            if in_repo.strip() and in_token.strip():
                cfg["github_repo"] = in_repo.strip()
                cfg["github_token"] = in_token.strip()
                save_config(cfg)
                storage._init_backend()
                # Test save
                ok, msg = storage.save_data(data)
                if ok:
                    st.success("🎉 GitHub와 성공적으로 연결되어 커밋이 완료되었습니다!")
                    st.rerun()
                else:
                    st.error(f"연결 오류: {msg}")
            else:
                st.warning("저장소 이름과 토큰을 모두 입력해주세요.")

    st.divider()
    st.caption(f"🕒 현재 기준 시각: **{now.strftime('%Y-%m-%d %H:%M')} (KST)**")

def parse_meal_date(date_str):
    m = re.search(r'(\d+)월\s*(\d+)일', date_str)
    if m:
        return datetime(current_year, int(m.group(1)), int(m.group(2))).date()
    return None

# Check if meal is eaten considering auto-deduction
def get_meal_eaten_status(meal, slot_type, auto_mode=False):
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

# ==========================================
# 👶 이유식 재료 조합 & 영양/궁합 분석 엔진
# ==========================================
NITRATE_VEG = {"청경채", "배추", "시금치", "비트", "근대", "상추", "상추류"}
CRUCIFEROUS_VEG = {"브로콜리", "양배추", "배추", "콜리플라워", "케일", "적채"}
GENTLE_VEG = {"애호박", "감자", "당근", "양파", "단호박"}

DISCOURAGED_PAIRS = [
    ({"단호박", "무"}, "단호박 + 무 (비타민 C 산화 및 소화 흡수 비효율)"),
    ({"오이", "당근"}, "오이 + 당근 (오이의 아스코르비나아제가 비타민C를 산화시킴)"),
    ({"감자", "고구마"}, "감자 + 고구마 (전분질 탄수화물 과다 & 가스 부담)"),
    ({"애호박", "단호박"}, "애호박 + 단호박 (유사 호박류 중복)"),
    ({"시금치", "두부"}, "시금치 + 두부 (옥살산과 칼슘이 결합하여 흡수 방해)"),
    ({"시금치", "멸치"}, "시금치 + 멸치 (옥살산-칼슘 흡수 방해)"),
    ({"시금치", "치즈"}, "시금치 + 치즈 (옥살산-칼슘 흡수 방해)"),
    ({"시금치", "비트"}, "시금치 + 비트 (질산염 및 옥살산 과다)"),
    ({"근대", "두부"}, "근대 + 두부 (옥살산-칼슘 흡수 방해)"),
    ({"브로콜리", "양배추"}, "브로콜리 + 양배추 (십자화과 중복, 가스 발생 가능)"),
    ({"브로콜리", "배추"}, "브로콜리 + 배추 (십자화과 중복, 가스 발생 가능)"),
    ({"양배추", "배추"}, "양배추 + 배추 (십자화과 중복, 가스 발생 가능)"),
    ({"브로콜리", "콜리플라워"}, "브로콜리 + 콜리플라워 (십자화과 중복, 가스 발생 가능)"),
]

GOLDEN_COMBOS = [
    ({"소고기", "브로콜리", "감자"}, "🥩 소고기 + 브로콜리 + 감자: 비타민C가 비헴철 흡수를 돕는 최고 궁합!"),
    ({"소고기", "브로콜리", "애호박"}, "🥩 소고기 + 브로콜리 + 애호박: 철분 흡수 극대화 & 소화 편안한 골든 조합!"),
    ({"소고기", "애호박", "당근"}, "🥩 소고기 + 애호박 + 당근: 균형 잡힌 영양 & 편안한 소화 추천!"),
    ({"소고기", "감자", "애호박"}, "🥩 소고기 + 감자 + 애호박: 부드러운 전분질과 담백한 소화 궁합!"),
    ({"애호박", "감자", "당근"}, "🥕 애호박 + 감자 + 당근: 실패 없는 순한 채소 삼총사 (소화 편안함 최고)!"),
    ({"애호박", "감자", "양파"}, "🧅 애호박 + 감자 + 양파: 풍미와 부드러움을 모두 잡은 최고 채소 조합!"),
    ({"애호박", "당근", "양파"}, "🧅 애호박 + 당근 + 양파: 달큰하고 향긋한 순한 채소 황금 조합!"),
    ({"감자", "당근", "양파"}, "🥔 감자 + 당근 + 양파: 아기들이 가장 좋아하는 기본 영양 채소 조합!"),
    ({"닭고기", "애호박", "당근"}, "🍗 닭고기 + 애호박 + 당근: 닭고기의 담백함과 순한 채소의 환상 궁합!"),
    ({"닭고기", "감자", "양파"}, "🍗 닭고기 + 감자 + 양파: 닭고기와 감자양파의 부드러운 스튜형 최고 궁합!"),
    ({"흰살생선", "감자", "애호박"}, "🐟 흰살생선 + 감자 + 애호박: 비린내 없이 담백하고 소화 잘 되는 조합!"),
    ({"당근", "애호박", "브로콜리"}, "🥦 당근 + 애호박 + 브로콜리: 비타민과 식이섬유가 균형 잡힌 추천 채소 조합!"),
    ({"감자", "애호박", "브로콜리"}, "🥦 감자 + 애호박 + 브로콜리: 부드러운 질감과 비타민C 충전 추천 조합!"),
]

def evaluate_meal_combo(ingredients):
    ing_set = set(ingredients)
    alerts = []
    goldens = []
    
    cruc_in_meal = ing_set.intersection(CRUCIFEROUS_VEG)
    has_cruc_alert = False
    if len(cruc_in_meal) >= 2:
        names = "+".join(cruc_in_meal)
        alerts.append(f"가스 주의: 십자화과 중복 ({names})")
        has_cruc_alert = True
        
    nitrate_in_meal = ing_set.intersection(NITRATE_VEG)
    has_nitrate_alert = False
    if len(nitrate_in_meal) >= 2:
        names = "+".join(nitrate_in_meal)
        alerts.append(f"질산염 주의: 질산염 채소 중복 ({names})")
        has_nitrate_alert = True
        
    for pair, reason in DISCOURAGED_PAIRS:
        if pair.issubset(ing_set):
            # Avoid duplicate warnings if cruciferous or nitrate is already flagged
            if has_cruc_alert and pair.issubset(CRUCIFEROUS_VEG):
                continue
            if has_nitrate_alert and pair.issubset(NITRATE_VEG):
                continue
            alerts.append(f"조합 주의: {reason.split(' (')[0]}")
            
    for combo, desc in GOLDEN_COMBOS:
        if combo.issubset(ing_set):
            goldens.append(desc)
            
    return {"alerts": alerts, "goldens": goldens}

def simulate_topping_addition(current_ingredients, candidate):
    new_list = list(current_ingredients) + [candidate]
    res = evaluate_meal_combo(new_list)
    curr_res = evaluate_meal_combo(current_ingredients)
    
    new_alerts = [a for a in res["alerts"] if a not in curr_res["alerts"]]
    if new_alerts:
        return "warning", "⚠️ [궁합 주의] " + new_alerts[0]
        
    new_goldens = [g for g in res["goldens"] if g not in curr_res["goldens"]]
    if new_goldens:
        return "golden", "🌟 [황금 궁합 달성!] " + new_goldens[0]
        
    if candidate in GENTLE_VEG:
        return "good", "🌱 [순한 채소] 소화 편안한 추천 재료"
        
    return "neutral", "✅ 무난한 조합"

def get_best_topping_recommendation(current_ingredients, inventory, surplus_items):
    curr_set = set(current_ingredients)
    curr_res = evaluate_meal_combo(current_ingredients)
    surplus_map = {s["name"]: s["surplus"] for s in surplus_items}
    candidates = []
    
    for item in inventory:
        ing_name = item["name"]
        stock = item.get("current_stock", 0)
        if stock <= 0:
            continue
        if ing_name in curr_set:
            continue
            
        test_ings = list(current_ingredients) + [ing_name]
        test_res = evaluate_meal_combo(test_ings)
        
        # Must not trigger new alerts
        new_alerts = [a for a in test_res["alerts"] if a not in curr_res["alerts"]]
        if new_alerts:
            continue
            
        new_goldens = [g for g in test_res["goldens"] if g not in curr_res["goldens"]]
        score = 0
        rec_type = "safe"
        rec_desc = ""
        
        if new_goldens:
            score += 120
            rec_type = "golden"
            rec_desc = "황금 궁합 완성: " + new_goldens[0].split(":")[0].replace("🥩 ", "").replace("🥕 ", "").replace("🍗 ", "").replace("🧅 ", "").replace("🥦 ", "").replace("🐟 ", "")
        elif ing_name in GENTLE_VEG:
            score += 60
            rec_type = "gentle"
            rec_desc = "소화 편안한 순한 채소"
        else:
            score += 20
            rec_type = "safe"
            rec_desc = "궁합 좋은 안전한 재료"
            
        # Surplus bonus: heavily prioritize surplus cubes!
        if ing_name in surplus_map:
            score += min(surplus_map[ing_name] * 8, 40)
            
        score += min(stock, 15)
        
        candidates.append({
            "name": ing_name,
            "category": item.get("category", "채소류"),
            "stock": stock,
            "surplus": surplus_map.get(ing_name, 0),
            "score": score,
            "type": rec_type,
            "desc": rec_desc
        })
        
    candidates.sort(key=lambda x: -x["score"])
    return candidates[0] if candidates else None

# Calculate Chronological Depletion & Deadlines with Automatic Deduction
def calculate_system_state(data, auto_mode=False):
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
                depletion_desc = f"✅ 현재 계획({len(meals)}일) 내 소진 안 됨 (잔여 {calc_stock - future_need}개 여유)"
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

        # Depletion timing score for sorting (lower score = depleted sooner)
        if calc_stock == 0 and future_need > 0:
            depletion_rank = -1000
        elif depletion_slot:
            depletion_rank = depletion_slot["slot_idx"]
        elif first_deficit_slot:
            depletion_rank = first_deficit_slot["slot_idx"]
        elif future_need > 0:
            depletion_rank = 10000 + (calc_stock - future_need)
        else:
            depletion_rank = 50000 - calc_stock if calc_stock > 0 else 90000

        timeline_results[name] = {
            "stock": calc_stock,
            "future_need": future_need,
            "net": calc_stock - future_need,
            "depletion_desc": depletion_desc,
            "depletion_slot": depletion_slot,
            "depletion_rank": depletion_rank
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
col_title, col_top_btn = st.columns([3, 1])
with col_title:
    st.markdown("<h2 style='margin:0; padding:4px 0; font-size:1.45rem;'>🥣 아기 이유식 & 큐브 플래너</h2>", unsafe_allow_html=True)
with col_top_btn:
    top_label = "🐙 Git 저장" if storage.mode == "github" else "☁️ 저장"
    if st.button(top_label, key="top_quick_save", use_container_width=True):
        with st.spinner("저장 중..."):
            ok, msg = storage.save_data(data)
        if ok:
            st.toast("✅ 실시간 저장 완료!")
            st.success(f"✅ 저장 완료 ({get_now().strftime('%H:%M:%S')})")
        else:
            st.error(f"❌ 실패: {msg}")

if "auto_deduct" not in st.session_state:
    st.session_state.auto_deduct = False

timeline_results, urgent_shortages, consumed_counts, surplus_items = calculate_system_state(data, auto_mode=st.session_state.auto_deduct)
inv_names = sorted([item["name"] for item in data.get("inventory", [])])

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🍽️ 오늘 식단",
    "🧊 큐브 제작",
    "📊 큐브 재고",
    "🛒 장보기",
    "📅 식단표",
    "🥗 영양 가이드"
])

# ==========================================
# TAB 1: 오늘의 식단 & 스마트 관리 (모바일 최적화)
# ==========================================
with tab1:
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

        # Top Bar: Date picker & Auto deduct switch
        c_date_sel, c_toggle = st.columns([3, 1])
        with c_date_sel:
            selected_date_str = st.selectbox("📅 날짜 선택", date_options, index=default_idx, label_visibility="collapsed")
            selected_idx = date_options.index(selected_date_str)
            meal_entry = meals[selected_idx]

        with c_toggle:
            st.session_state.auto_deduct = st.toggle("🤖 자동 차감", value=st.session_state.auto_deduct, help="기본값은 '수동 차감(OFF)'입니다. 켜두시면 10시·14시·19시 시간 경과 시 냉동실 재고가 자동 차감됩니다.")

        if meal_entry.get("note"):
            st.caption(f"💡 **특이사항**: {meal_entry['note']}")

        # 📱 Mobile-friendly Meal Segmented Switcher (Default to current time of day)
        curr_hour = now.hour
        default_slot_choice = 0 if curr_hour < 11 else (1 if curr_hour < 16 else 2)
        
        c_slot_sel, c_view_all = st.columns([3, 1])
        with c_slot_sel:
            chosen_slot = st.radio(
                "끼니 선택",
                ["🌅 아침", "☀️ 점심", "🌙 저녁"],
                index=default_slot_choice,
                horizontal=True,
                label_visibility="collapsed",
                key="mobile_meal_slot_radio"
            )
        with c_view_all:
            show_all_three = st.checkbox("세 끼 전체", value=False, key="chk_show_all_three", help="모바일에서 한 번에 3끼 모두 보려면 체크하세요.")

        def set_override(meal_obj, slot_type, override_val):
            meal_obj[f"{slot_type}_override"] = override_val
            storage.save_data(data)
            st.rerun()

        # Slot card renderer with compact styling and one-click topping recommendation
        def render_meal_slot_card(slot_key, slot_name, slot_short, meal_obj):
            ings = meal_obj.get(slot_key, [])
            is_eaten, status_text = get_meal_eaten_status(meal_obj, slot_key, st.session_state.auto_deduct)
            
            if is_eaten:
                st.markdown(f"""
                <div style="background:#ecfdf5; border:1.5px solid #10b981; border-radius:10px; padding:6px 12px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:1.1rem; font-weight:bold; color:#065f46;">{slot_name}</span>
                    <span style="background:#10b981; color:white; font-size:0.75rem; font-weight:bold; padding:2px 8px; border-radius:12px;">{status_text}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:6px 12px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:1.1rem; font-weight:bold; color:#1e293b;">{slot_name}</span>
                    <span style="background:#f1f5f9; color:#64748b; font-size:0.75rem; font-weight:600; padding:2px 8px; border-radius:12px; border:1px solid #e2e8f0;">⏳ 식사 전</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Ingredient tags
            ing_html = "".join([f'<span class="tag-badge {get_badge_class(get_ingredient_category(x))}">{x}</span>' for x in ings])
            st.markdown(ing_html, unsafe_allow_html=True)
            
            # Subtle compact pill tags for Golden / Alert combos
            combo_res = evaluate_meal_combo(ings)
            pill_html = ""
            for g in combo_res["goldens"]:
                short_g = g.split(":")[0].strip()
                pill_html += f'<span style="display:inline-block; background:#ecfdf5; border:1px solid #a7f3d0; color:#065f46; border-radius:12px; padding:2px 7px; font-size:0.75rem; font-weight:600; margin-right:4px; margin-top:3px;">✨ {short_g}</span>'
            for a in combo_res["alerts"]:
                pill_html += f'<span style="display:inline-block; background:#fffbeb; border:1px solid #fde68a; color:#92400e; border-radius:12px; padding:2px 7px; font-size:0.75rem; font-weight:600; margin-right:4px; margin-top:3px;">⚠️ {a}</span>'
            
            if pill_html:
                st.markdown(f'<div style="margin: 3px 0 5px 0;">{pill_html}</div>', unsafe_allow_html=True)

            # 💡 [반찬 1개 더 추가 추천] - Compact 1-line bar
            rec = get_best_topping_recommendation(ings, data.get("inventory", []), surplus_items)
            if rec:
                surplus_tag = f"(+{rec['surplus']}개 잉여)" if rec["surplus"] > 0 else f"(재고 {rec['stock']}개)"
                badge_color = "#047857" if rec["type"] == "golden" else ("#0284c7" if rec["type"] == "gentle" else "#475569")
                
                c_rec_txt, c_rec_btn = st.columns([3, 1])
                with c_rec_txt:
                    st.markdown(f"""
                    <div style="background:#f8fafc; border:1px dashed #cbd5e1; border-radius:8px; padding:5px 8px; font-size:0.78rem; line-height:1.35;">
                        💡 <b>추천:</b> <b style="color:{badge_color};">{rec['name']}</b> <small style="color:#64748b;">{surplus_tag}</small><br>
                        <span style="color:#475569; font-size:0.72rem;">└ {rec['desc']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                with c_rec_btn:
                    st.write("")
                    if st.button(f"➕ {rec['name']}", key=f"quick_add_{slot_key}_{selected_idx}", help=f"{rec['name']} 큐브를 이 끼니에 추가합니다", use_container_width=True):
                        meal_obj[slot_key].append(rec["name"])
                        storage.save_data(data)
                        st.toast(f"🎉 {selected_date_str} {slot_name}에 '{rec['name']}' 추가 완료!")
                        st.rerun()

            # Action row: Status / Eat button + Manage popover in 1 balanced line
            c_act_btn, c_pop = st.columns([2, 1])
            with c_act_btn:
                if is_eaten:
                    if st.button(f"❌ {status_text} (취소)", key=f"skip_{slot_key}", use_container_width=True):
                        set_override(meal_obj, slot_key, "skipped")
                else:
                    if st.button(f"🥣 식사 완료 처리", key=f"force_eat_{slot_key}", type="primary", use_container_width=True):
                        set_override(meal_obj, slot_key, "eaten")

            with c_pop:
                with st.popover("✏️ 재료 관리", use_container_width=True):
                    st.caption(f"{slot_name} 재료 목록")
                    current_ings = list(meal_obj.get(slot_key, []))
                    for ing in current_ings:
                        c_txt, c_del = st.columns([3, 1])
                        c_txt.write(f"• {ing}")
                        if c_del.button("삭제", key=f"del_{slot_key}_{selected_idx}_{ing}"):
                            meal_obj[slot_key].remove(ing)
                            storage.save_data(data)
                            st.rerun()

                    st.divider()
                    st.caption("➕ 재료 직접 추가")
                    available_to_add = [n for n in inv_names if n not in current_ings]
                    if available_to_add:
                        c_sel, c_add = st.columns([2, 1])
                        with c_sel:
                            add_ing = st.selectbox("재료 선택", available_to_add, key=f"sel_add_{slot_key}_{selected_idx}", label_visibility="collapsed")
                        with c_add:
                            if st.button("추가", key=f"btn_manual_add_{slot_key}_{selected_idx}", use_container_width=True):
                                if slot_key not in meal_obj:
                                    meal_obj[slot_key] = []
                                meal_obj[slot_key].append(add_ing)
                                storage.save_data(data)
                                st.toast(f"✅ {slot_name}에 '{add_ing}' 추가 완료!")
                                st.rerun()

        # Render Meals based on View Mode
        if show_all_three:
            col_m, col_l, col_d = st.columns(3)
            with col_m:
                render_meal_slot_card("morning", "🌅 아침 식단", "아침", meal_entry)
            with col_l:
                render_meal_slot_card("lunch", "☀️ 점심 식단", "점심", meal_entry)
            with col_d:
                render_meal_slot_card("dinner", "🌙 저녁 식단", "저녁", meal_entry)
        else:
            slot_info_map = {
                "🌅 아침": ("morning", "🌅 아침 식단", "아침"),
                "☀️ 점심": ("lunch", "☀️ 점심 식단", "점심"),
                "🌙 저녁": ("dinner", "🌙 저녁 식단", "저녁"),
            }
            s_key, s_name, s_short = slot_info_map[chosen_slot]
            render_meal_slot_card(s_key, s_name, s_short, meal_entry)

        # ==========================================
        # 💡 [냉동실 털기] 접이식 Expander로 정리하여 화면 깔끔화
        # ==========================================
        st.write("")
        with st.expander("💡 [냉동실 털기] 남아도는 잉여 큐브 소진 & 추가 (선택사항)", expanded=False):
            st.caption("남은 15일 식단에 계획된 수량보다 냉동실에 더 많이 남아있는 **잉여 큐브**를 원하는 끼니에 추가합니다.")
            if not surplus_items:
                st.success("🎉 현재 냉동실에 남아도는 잉여 큐브가 없습니다!")
            else:
                surplus_html = " &nbsp;|&nbsp; ".join([
                    f"<b>{s['name']}</b>: <b style='color:#ef4444;'>+{s['surplus']}개</b> (잔여 {s['stock']}개)"
                    for s in surplus_items[:6]
                ])
                st.markdown(f"""
                <div style="background-color:#fff7ed; border:1px solid #fed7aa; border-radius:8px; padding:8px 12px; margin-bottom:10px; font-size:0.83rem;">
                    📢 <b>잉여 큐브:</b> {surplus_html}
                </div>
                """, unsafe_allow_html=True)

                c_add_slot, c_add_item, c_add_btn = st.columns([1, 2, 1])
                slot_map = {"🌅 아침": "morning", "☀️ 점심": "lunch", "🌙 저녁": "dinner"}
                with c_add_slot:
                    target_slot = st.selectbox("끼니", ["🌅 아침", "☀️ 점심", "🌙 저녁"], key="surplus_slot_choice")
                    target_slot_key = slot_map[target_slot]
                    target_current_ings = meal_entry.get(target_slot_key, [])

                with c_add_item:
                    surplus_options = []
                    opt_info_map = {}
                    for s in surplus_items:
                        s_name = s["name"]
                        score_type, sim_msg = simulate_topping_addition(target_current_ings, s_name)
                        tag = "🌟최고궁합" if score_type == "golden" else ("⚠️가스/주의" if score_type == "warning" else ("🌱순한채소" if score_type == "good" else "✅무난"))
                        opt_label = f"{s_name} (+{s['surplus']}개) [{tag}]"
                        surplus_options.append(opt_label)
                        opt_info_map[opt_label] = (s_name, score_type, sim_msg)

                    chosen_opt = st.selectbox("추가할 잉여 큐브", surplus_options, key="surplus_item_choice")
                    chosen_ing_name, chosen_score, chosen_msg = opt_info_map[chosen_opt]

                with c_add_btn:
                    st.write("")
                    if st.button("➕ 추가", type="primary", use_container_width=True, key="btn_add_surplus"):
                        slot_key = slot_map[target_slot]
                        if chosen_ing_name in meal_entry.get(slot_key, []):
                            st.warning(f"이미 포함되어 있습니다!")
                        else:
                            if slot_key not in meal_entry:
                                meal_entry[slot_key] = []
                            meal_entry[slot_key].append(chosen_ing_name)
                            storage.save_data(data)
                            st.toast(f"🎉 '{chosen_ing_name}' 큐브 추가 완료!")
                            st.rerun()

# ==========================================
# TAB 2: 큐브 제작 & 입고 등록
# ==========================================
with tab2:
    st.subheader("🧊 큐브 새로 만들었을 때 바로 입고 등록")
    st.caption("새로 만든 큐브를 등록하면 냉동실 재고, 예상 소진일, 마감일 알림이 구글 시트와 클라우드에 즉시 저장됩니다.")

    inv_names = [item["name"] for item in data.get("inventory", [])]
    inv_names.sort()

    c1, c2 = st.columns(2)
    with c1:
        sel_mode = st.radio("재료 선택 방식", ["기존 재료 선택", "➕ 새 재료 직접 추가"], horizontal=True, key="cube_sel_mode")
        if sel_mode == "기존 재료 선택":
            cube_ingredient = st.selectbox("품목 (재료)", inv_names, key="cube_sel_ing")
            cube_category = get_ingredient_category(cube_ingredient)
        else:
            cube_ingredient = st.text_input("새 재료 이름 (예: 콜리플라워, 비트)", key="cube_custom_ing")
            cube_category = st.selectbox("분류", ["채소류", "단백질", "곡류", "과일/기타"], key="cube_custom_cat")

    with c2:
        cube_qty = st.number_input("제작 수량 (개)", min_value=1, max_value=60, value=12, step=1, key="cube_qty_input")
        today_date = now.date()
        make_date = st.date_input("제작일", value=today_date, key="cube_make_date")

        # Dynamic default 2 weeks (14 days) based on make_date; manual override preserved
        if "prev_cube_make_date" not in st.session_state or st.session_state["prev_cube_make_date"] != make_date:
            st.session_state["prev_cube_make_date"] = make_date
            st.session_state["cube_exp_date"] = make_date + timedelta(days=14)

        exp_date = st.date_input(
            "권장 소비기한 (제작일 + 2주 기본 권장)",
            key="cube_exp_date",
            help="이유식 큐브는 신선도와 영양 보존을 위해 냉동 보관 2주 이내 소비를 권장합니다. 필요한 경우 날짜를 직접 변경하실 수 있습니다."
        )

    memo = st.text_input("메모 (선택사항)", placeholder="예: 15g 큐브 12구 1판, 무항생제 닭안심 사용", key="cube_memo_input")

    submit_cube = st.button("🧊 냉동실 큐브 입고 등록 (+ 반영하기)", type="primary", use_container_width=True, key="btn_submit_cube")

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
                "memo": memo.strip()
            })

            storage.save_data(data)
            st.toast(f"🎉 [{ing_name}] 큐브 +{cube_qty}개가 성공적으로 입고되었습니다!")
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
    c_filter, c_sort = st.columns([3, 2])
    with c_filter:
        filter_cat = st.radio("📂 카테고리 필터", ["전체", "단백질", "채소류", "곡류"], horizontal=True, key="tab3_filter_cat")
    with c_sort:
        sort_by = st.selectbox("🔄 정렬 기준", [
            "⏰ 소진시점 빠른 순 (임박/부족순 ⭐)",
            "⏰ 소진시점 여유 있는 순",
            "🧊 실재고 적은 순 (0개 우선)",
            "🧊 실재고 많은 순",
            "⌛ 권장 소비기한 임박순 (제작 2주 기준)",
            "🏷️ 가나다 이름순"
        ], key="tab3_sort_by")

    # Map of latest production logs per ingredient
    latest_logs = {}
    for log in data.get("production_logs", []):
        ing = log.get("ingredient")
        if ing and ing not in latest_logs:
            latest_logs[ing] = log

    filtered_inv = [item for item in inv if filter_cat == "전체" or item["category"] == filter_cat]

    if sort_by == "⏰ 소진시점 빠른 순 (임박/부족순 ⭐)":
        filtered_inv.sort(key=lambda x: (
            timeline_results.get(x["name"], {}).get("depletion_rank", 99999),
            -timeline_results.get(x["name"], {}).get("future_need", 0),
            x["name"]
        ))
    elif sort_by == "⏰ 소진시점 여유 있는 순":
        filtered_inv.sort(key=lambda x: (
            -timeline_results.get(x["name"], {}).get("depletion_rank", 99999),
            -timeline_results.get(x["name"], {}).get("stock", 0),
            x["name"]
        ))
    elif sort_by == "🧊 실재고 적은 순 (0개 우선)":
        filtered_inv.sort(key=lambda x: (
            timeline_results.get(x["name"], {}).get("stock", 0),
            -timeline_results.get(x["name"], {}).get("future_need", 0),
            x["name"]
        ))
    elif sort_by == "🧊 실재고 많은 순":
        filtered_inv.sort(key=lambda x: (
            -timeline_results.get(x["name"], {}).get("stock", 0),
            x["name"]
        ))
    elif sort_by == "⌛ 권장 소비기한 임박순 (제작 2주 기준)":
        def exp_sort_key(item):
            log = latest_logs.get(item["name"])
            if log and log.get("exp_date"):
                try:
                    return (0, datetime.strptime(log["exp_date"], "%Y-%m-%d").date())
                except:
                    pass
            return (1, date(9999, 12, 31))
        filtered_inv.sort(key=exp_sort_key)
    elif sort_by == "🏷️ 가나다 이름순":
        filtered_inv.sort(key=lambda x: x["name"])

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

        # Check latest production / expiry info
        log_info_html = ""
        log = latest_logs.get(name)
        if log:
            exp_str = log.get("exp_date", "")
            make_str = log.get("date", "")
            exp_badge = ""
            if exp_str:
                try:
                    exp_d = datetime.strptime(exp_str, "%Y-%m-%d").date()
                    d_left = (exp_d - now.date()).days
                    if d_left < 0:
                        exp_badge = f'<span style="color:#ef4444; font-weight:bold; background:#fee2e2; border:1px solid #fca5a5; padding:1px 6px; border-radius:4px; font-size:0.75rem;">🚨 소비기한 경과 ({abs(d_left)}일 전)</span>'
                    elif d_left <= 3:
                        exp_badge = f'<span style="color:#d97706; font-weight:bold; background:#fef3c7; border:1px solid #fcd34d; padding:1px 6px; border-radius:4px; font-size:0.75rem;">⏳ 소비기한 임박 (D-{d_left})</span>'
                    else:
                        exp_badge = f'<span style="color:#059669; font-weight:600; background:#ecfdf5; border:1px solid #a7f3d0; padding:1px 6px; border-radius:4px; font-size:0.75rem;">D-{d_left} 남음</span>'
                except:
                    pass
            log_info_html = f"""
            <div style="margin-top:5px; padding:4px 8px; background:rgba(255,255,255,0.65); border-radius:6px; font-size:0.83rem; color:#475569; display:flex; justify-content:space-between; align-items:center;">
                <span>🧊 <b>최근 제작:</b> {make_str} | ⌛ <b>권장 소비기한(2주):</b> {exp_str}</span>
                <span>{exp_badge}</span>
            </div>
            """

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
                {log_info_html}
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
# TAB 5: 이유식 식단표 전체보기 & 식단 관리
# ==========================================
with tab5:
    meals_list = data.get("meals", [])
    st.subheader(f"📅 이유식 식단표 (총 {len(meals_list)}일차)")
    
    # 📋 1. 특정 날짜 식단 그대로 복제하기
    with st.expander("📋 특정 날짜 식단 그대로 복제하기 (1일 또는 2~3일 연속 복제)", expanded=False):
        st.markdown("이미 작성된 날짜의 식단(아침·점심·저녁 큐브 및 메모)을 다른 날짜에 **그대로 복사**하여 등록합니다.")
        if not meals_list:
            st.warning("복사할 기존 식단이 없습니다.")
        else:
            c_src, c_tgt = st.columns(2)
            with c_src:
                copy_src_options = [f"{idx+1}일차: {m['date']} ({m['day_of_week']})" for idx, m in enumerate(meals_list)]
                default_src_idx = len(copy_src_options) - 1
                sel_src_label = st.selectbox("1️⃣ 복사할 원본 날짜 선택", copy_src_options, index=default_src_idx, key="copy_src_date_sel")
                src_idx = copy_src_options.index(sel_src_label)
                src_meal = meals_list[src_idx]
                
                m_str = ', '.join(src_meal.get('morning', [])) or '(없음)'
                l_str = ', '.join(src_meal.get('lunch', [])) or '(없음)'
                d_str = ', '.join(src_meal.get('dinner', [])) or '(없음)'
                nt_str = src_meal.get('note', '-')
                st.markdown(f"""
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:10px; font-size:0.83rem; line-height:1.45; margin-top:5px;">
                    <b>🌅 아침:</b> {m_str}<br>
                    <b>☀️ 점심:</b> {l_str}<br>
                    <b>🌙 저녁:</b> {d_str}<br>
                    <small style="color:#64748b;">💡 메모: {nt_str}</small>
                </div>
                """, unsafe_allow_html=True)

            with c_tgt:
                default_copy_target = now.date()
                last_d = parse_meal_date(meals_list[-1]["date"])
                if last_d:
                    default_copy_target = last_d + timedelta(days=1)
                
                copy_start_date = st.date_input("2️⃣ 적용할 새 시작 날짜", value=default_copy_target, key="copy_target_date_input")
                repeat_days = st.radio("3️⃣ 적용 일수 (이유식 큐브 주기)", [1, 2, 3], index=0, horizontal=True,
                                       format_func=lambda x: f"{x}일간 동일 적용 (큐브 묶음)" if x > 1 else "1일만 복사", key="copy_repeat_days")
                
                date_preview_list = []
                for off in range(repeat_days):
                    dt = copy_start_date + timedelta(days=off)
                    dow = ["월", "화", "수", "목", "금", "토", "일"][dt.weekday()]
                    date_preview_list.append(f"{dt.month}월 {dt.day}일 ({dow})")
                
                st.caption(f"🗓️ 복사 적용 예정: **{', '.join(date_preview_list)}**")

            st.write("")
            if st.button(f"📋 {src_meal['date']} 식단을 {repeat_days}일간 복제 등록하기", type="primary", use_container_width=True, key="btn_execute_copy_meal"):
                KOREAN_DAYS = ["월", "화", "수", "목", "금", "토", "일"]
                for off in range(repeat_days):
                    dt = copy_start_date + timedelta(days=off)
                    t_str = f"{dt.month}월 {dt.day}일"
                    t_dow = KOREAN_DAYS[dt.weekday()]
                    
                    exist_i = None
                    for idx, m in enumerate(data.get("meals", [])):
                        if m["date"].replace(" ", "") == t_str.replace(" ", ""):
                            exist_i = idx
                            break
                    
                    new_m = {
                        "date": t_str,
                        "day_of_week": t_dow,
                        "morning": list(src_meal.get("morning", [])),
                        "lunch": list(src_meal.get("lunch", [])),
                        "dinner": list(src_meal.get("dinner", [])),
                        "morning_eaten": False,
                        "lunch_eaten": False,
                        "dinner_eaten": False,
                        "note": src_meal.get("note", "")
                    }
                    if "meals" not in data:
                        data["meals"] = []
                    
                    if exist_i is not None:
                        data["meals"][exist_i] = new_m
                    else:
                        data["meals"].append(new_m)

                def sort_meal_key(m):
                    d = parse_meal_date(m.get("date", ""))
                    return d if d else date(9999, 12, 31)
                data["meals"].sort(key=sort_meal_key)
                
                storage.save_data(data)
                st.toast(f"🎉 {src_meal['date']} 식단이 {repeat_days}일간 성공적으로 복제되었습니다!")
                st.rerun()

    # ➕ 2. 새로운 날짜 식단 직접 구성하기
    with st.expander("➕ 새로운 날짜 식단 직접 구성하기 (재료 맞춤 등록)", expanded=False):
        st.markdown("새로운 날짜의 식단을 등록하면, **냉동실 큐브 소진 시점과 장보기 일정**이 즉시 자동으로 계산됩니다.")
        
        # Helper: Load from template
        c_tmpl, c_tmpl_btn = st.columns([3, 1])
        with c_tmpl:
            tmpl_options = ["(새로 직접 선택)"] + [f"{idx+1}일차: {m['date']} ({m['day_of_week']})" for idx, m in enumerate(meals_list)]
            sel_tmpl = st.selectbox("📋 기존 날짜 식단 불러와서 채우기 (선택)", tmpl_options, key="sel_tmpl_box")
        with c_tmpl_btn:
            st.write("")
            if st.button("📥 불러오기", key="btn_apply_tmpl", use_container_width=True):
                if sel_tmpl != "(새로 직접 선택)":
                    t_idx = tmpl_options.index(sel_tmpl) - 1
                    chosen_m = meals_list[t_idx]
                    st.session_state["new_m_ings_sel"] = [x for x in chosen_m.get("morning", []) if x in inv_names]
                    st.session_state["new_l_ings_sel"] = [x for x in chosen_m.get("lunch", []) if x in inv_names]
                    st.session_state["new_d_ings_sel"] = [x for x in chosen_m.get("dinner", []) if x in inv_names]
                    st.session_state["new_meal_note_input"] = chosen_m.get("note", "")
                    st.toast(f"✅ {chosen_m['date']} 식단을 불러왔습니다! 필요 시 수정 후 저장하세요.")
                    st.rerun()

        # Calculate default date (next day after the last registered meal)
        default_new_date = now.date()
        if meals_list:
            last_d = parse_meal_date(meals_list[-1]["date"])
            if last_d:
                default_new_date = last_d + timedelta(days=1)

        c_dt, c_nt = st.columns([1, 2])
        with c_dt:
            new_date_val = st.date_input("식단 날짜 선택", value=default_new_date, key="new_meal_date_input")
            k_dow = ["월", "화", "수", "목", "금", "토", "일"][new_date_val.weekday()]
            new_date_str = f"{new_date_val.month}월 {new_date_val.day}일"
            st.caption(f"등록 날짜: **{new_date_str} ({k_dow})**")
        with c_nt:
            if "new_meal_note_input" not in st.session_state:
                st.session_state["new_meal_note_input"] = ""
            new_note = st.text_input("특이사항 / 메모 (선택)", placeholder="예: 소고기 증량 시작, 첫 생선 테스트 등", key="new_meal_note_input")

        st.caption("🥣 끼니별 큐브 재료를 선택하세요 (선택 시 황금 궁합/주의 알림이 즉시 표시됩니다)")
        
        if "new_m_ings_sel" not in st.session_state:
            st.session_state["new_m_ings_sel"] = [x for x in ["쌀죽", "소고기"] if x in inv_names]
        if "new_l_ings_sel" not in st.session_state:
            st.session_state["new_l_ings_sel"] = [x for x in ["쌀죽", "닭고기"] if x in inv_names]
        if "new_d_ings_sel" not in st.session_state:
            st.session_state["new_d_ings_sel"] = [x for x in ["쌀죽"] if x in inv_names]

        c_m, c_l, c_d = st.columns(3)
        with c_m:
            st.markdown("##### 🌅 아침")
            new_m_ings = st.multiselect("아침 재료", inv_names, key="new_m_ings_sel", label_visibility="collapsed")
            m_comb = evaluate_meal_combo(new_m_ings)
            for g in m_comb["goldens"]:
                st.caption(f"✨ {g.split(':')[0]}")
            for a in m_comb["alerts"]:
                st.caption(f"⚠️ {a}")

        with c_l:
            st.markdown("##### ☀️ 점심")
            new_l_ings = st.multiselect("점심 재료", inv_names, key="new_l_ings_sel", label_visibility="collapsed")
            l_comb = evaluate_meal_combo(new_l_ings)
            for g in l_comb["goldens"]:
                st.caption(f"✨ {g.split(':')[0]}")
            for a in l_comb["alerts"]:
                st.caption(f"⚠️ {a}")

        with c_d:
            st.markdown("##### 🌙 저녁")
            new_d_ings = st.multiselect("저녁 재료", inv_names, key="new_d_ings_sel", label_visibility="collapsed")
            d_comb = evaluate_meal_combo(new_d_ings)
            for g in d_comb["goldens"]:
                st.caption(f"✨ {g.split(':')[0]}")
            for a in d_comb["alerts"]:
                st.caption(f"⚠️ {a}")

        st.write("")
        if st.button("➕ 이 날짜 식단을 식단표에 추가 및 저장", type="primary", use_container_width=True, key="btn_save_new_meal"):
            if not (new_m_ings or new_l_ings or new_d_ings):
                st.warning("최소 한 끼 이상의 재료를 선택해주세요.")
            else:
                existing_idx = None
                for idx, m in enumerate(data.get("meals", [])):
                    if m["date"].replace(" ", "") == new_date_str.replace(" ", ""):
                        existing_idx = idx
                        break

                new_entry = {
                    "date": new_date_str,
                    "day_of_week": k_dow,
                    "morning": new_m_ings,
                    "lunch": new_l_ings,
                    "dinner": new_d_ings,
                    "morning_eaten": False,
                    "lunch_eaten": False,
                    "dinner_eaten": False,
                    "note": new_note.strip()
                }
                if "meals" not in data:
                    data["meals"] = []
                
                if existing_idx is not None:
                    data["meals"][existing_idx] = new_entry
                    st.toast(f"🔄 {new_date_str} 기존 식단이 업데이트되었습니다!")
                else:
                    data["meals"].append(new_entry)
                    def sort_meal_key(m):
                        d = parse_meal_date(m.get("date", ""))
                        return d if d else date(9999, 12, 31)
                    data["meals"].sort(key=sort_meal_key)
                    st.toast(f"🎉 {new_date_str} ({k_dow}) 식단이 새로 추가되었습니다!")
                
                storage.save_data(data)
                st.rerun()

    # 🛠️ 2. 등록된 식단 날짜 수정 / 삭제
    if meals_list:
        with st.expander("🛠️ 등록된 식단 날짜 수정 / 삭제", expanded=False):
            edit_options = [f"{idx+1}일차: {m['date']} ({m['day_of_week']})" for idx, m in enumerate(meals_list)]
            sel_edit_label = st.selectbox("수정 또는 삭제할 식단 날짜 선택", edit_options, key="sel_edit_meal_idx")
            edit_target_idx = edit_options.index(sel_edit_label)
            target_m = meals_list[edit_target_idx]

            all_choices = sorted(list(set(inv_names + target_m.get("morning", []) + target_m.get("lunch", []) + target_m.get("dinner", []))))
            
            c_ed_m, c_ed_l, c_ed_d = st.columns(3)
            with c_ed_m:
                ed_m = st.multiselect("🌅 아침 재료 수정", all_choices, default=[x for x in target_m.get("morning", []) if x in all_choices], key=f"ed_m_{edit_target_idx}")
            with c_ed_l:
                ed_l = st.multiselect("☀️ 점심 재료 수정", all_choices, default=[x for x in target_m.get("lunch", []) if x in all_choices], key=f"ed_l_{edit_target_idx}")
            with c_ed_d:
                ed_d = st.multiselect("🌙 저녁 재료 수정", all_choices, default=[x for x in target_m.get("dinner", []) if x in all_choices], key=f"ed_d_{edit_target_idx}")

            ed_note = st.text_input("메모 / 특이사항 수정", value=target_m.get("note", ""), key=f"ed_note_{edit_target_idx}")

            c_save_ed, c_del_ed = st.columns([2, 1])
            with c_save_ed:
                if st.button("💾 식단 수정 저장", type="primary", use_container_width=True, key=f"btn_save_ed_{edit_target_idx}"):
                    target_m["morning"] = ed_m
                    target_m["lunch"] = ed_l
                    target_m["dinner"] = ed_d
                    target_m["note"] = ed_note.strip()
                    storage.save_data(data)
                    st.toast(f"✅ {target_m['date']} 식단이 수정되었습니다!")
                    st.rerun()
            with c_del_ed:
                if st.button(f"🗑️ {target_m['date']} 식단 삭제", use_container_width=True, key=f"btn_del_ed_{edit_target_idx}"):
                    deleted_date = target_m['date']
                    data["meals"].pop(edit_target_idx)
                    storage.save_data(data)
                    st.toast(f"🗑️ {deleted_date} 식단이 삭제되었습니다!")
                    st.rerun()

    st.write("")
    today_str = f"{now.month}월{now.day}일"
    
    # Legend guide
    st.markdown("""
    <div style="display:flex; flex-wrap:wrap; gap:10px; align-items:center; font-size:0.83rem; margin-bottom:12px; background:#f8fafc; padding:8px 12px; border-radius:8px; border:1px solid #e2e8f0;">
        <span>🎨 <b>식단 색상 구분:</b></span>
        <span style="background:#dcfce7; border:1px solid #86efac; color:#166534; padding:2px 8px; border-radius:6px; font-weight:bold;">✅ 식사 완료 (연초록색 강조)</span>
        <span style="background:#ffffff; border:1px solid #e2e8f0; color:#64748b; padding:2px 8px; border-radius:6px;">⬜ 식사 대기 (미완료)</span>
        <span style="color:#64748b; font-size:0.78rem;">(💡 식사 완료 처리는 <b>[🍽️ 오늘 식단]</b> 탭에서 끼니별로 간편하게 하실 수 있습니다)</span>
    </div>
    """, unsafe_allow_html=True)

    def render_table_slot_cell(ings, is_eaten):
        if not ings:
            return '<td style="padding:10px 12px; color:#cbd5e1; font-size:0.82rem; border-bottom:1px solid #e2e8f0; text-align:center;">-</td>'
        ings_txt = ", ".join(ings)
        if is_eaten:
            return f'''<td style="padding:10px 12px; background:#dcfce7; border-bottom:1px solid #bbf7d0; border-left:1px solid #bbf7d0; color:#14532d;">
                <div style="margin-bottom:3px;">
                    <span style="display:inline-block; background:#16a34a; color:white; font-size:0.68rem; font-weight:bold; padding:1px 6px; border-radius:10px;">✅ 완료</span>
                </div>
                <div style="font-weight:600; font-size:0.84rem; line-height:1.4;">{ings_txt}</div>
            </td>'''
        else:
            return f'''<td style="padding:10px 12px; background:#ffffff; border-bottom:1px solid #f1f5f9; border-left:1px solid #f1f5f9; color:#334155;">
                <div style="margin-bottom:3px;">
                    <span style="display:inline-block; background:#f1f5f9; color:#64748b; font-size:0.68rem; padding:1px 6px; border-radius:10px; border:1px solid #e2e8f0;">대기</span>
                </div>
                <div style="font-size:0.84rem; line-height:1.4; color:#475569;">{ings_txt}</div>
            </td>'''

    html_rows = []
    table_rows = []
    for m in data.get("meals", []):
        m_eaten, _ = get_meal_eaten_status(m, "morning", st.session_state.auto_deduct)
        l_eaten, _ = get_meal_eaten_status(m, "lunch", st.session_state.auto_deduct)
        d_eaten, _ = get_meal_eaten_status(m, "dinner", st.session_state.auto_deduct)
        
        is_today = (m["date"].replace(" ", "") == today_str)
        all_eaten = (m_eaten and l_eaten and d_eaten)
        
        date_badge = ""
        if is_today:
            date_badge = '<br><span style="display:inline-block; background:#3b82f6; color:white; font-size:0.68rem; font-weight:bold; padding:1px 6px; border-radius:10px; margin-top:2px;">📍 오늘</span>'
        elif all_eaten:
            date_badge = '<br><span style="display:inline-block; background:#10b981; color:white; font-size:0.68rem; font-weight:bold; padding:1px 6px; border-radius:10px; margin-top:2px;">🎉 올클리어</span>'

        date_bg = "#eff6ff" if is_today else ("#f0fdf4" if all_eaten else "#f8fafc")
        
        row_html = f'''<tr>
            <td style="padding:10px 12px; background:{date_bg}; border-bottom:1px solid #e2e8f0; font-weight:bold; color:#1e293b; white-space:nowrap; vertical-align:middle;">
                {m["date"]} ({m["day_of_week"]}){date_badge}
            </td>
            {render_table_slot_cell(m.get("morning", []), m_eaten)}
            {render_table_slot_cell(m.get("lunch", []), l_eaten)}
            {render_table_slot_cell(m.get("dinner", []), d_eaten)}
            <td style="padding:10px 12px; background:#f8fafc; border-bottom:1px solid #e2e8f0; border-left:1px solid #f1f5f9; font-size:0.8rem; color:#64748b; vertical-align:middle;">
                {m.get("note", "-") or "-"}
            </td>
        </tr>'''
        html_rows.append(row_html)

        table_rows.append({
            "날짜": m["date"],
            "요일": m["day_of_week"],
            "아침": f"{'✅' if m_eaten else '⬜'} " + ", ".join(m.get("morning", [])),
            "점심": f"{'✅' if l_eaten else '⬜'} " + ", ".join(m.get("lunch", [])),
            "저녁": f"{'✅' if d_eaten else '⬜'} " + ", ".join(m.get("dinner", [])),
            "비고": m.get("note", "")
        })

    full_table_html = f'''
    <div style="overflow-x:auto; -webkit-overflow-scrolling:touch; border:1px solid #cbd5e1; border-radius:10px; margin-bottom:15px; box-shadow:0 1px 3px rgba(0,0,0,0.06);">
    <table style="width:100%; border-collapse:collapse; font-size:0.85rem; font-family:-apple-system, BlinkMacSystemFont, sans-serif; text-align:left; min-width:680px;">
      <thead>
        <tr style="background:#e2e8f0; border-bottom:2px solid #cbd5e1; color:#334155; font-size:0.86rem;">
          <th style="padding:10px 12px; width:15%;">📅 날짜 (요일)</th>
          <th style="padding:10px 12px; width:26%;">🌅 아침 식단</th>
          <th style="padding:10px 12px; width:26%;">☀️ 점심 식단</th>
          <th style="padding:10px 12px; width:26%;">🌙 저녁 식단</th>
          <th style="padding:10px 12px; width:7%;">비고</th>
        </tr>
      </thead>
      <tbody>
        {"".join(html_rows)}
      </tbody>
    </table>
    </div>
    '''
    st.markdown(full_table_html, unsafe_allow_html=True)

    with st.expander("📋 텍스트 표로 보기 (클립보드 복사용)", expanded=False):
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("💾 데이터 내보내기 & 영구 저장")
    c_save_sheet, c_save_excel = st.columns(2)

    with c_save_sheet:
        save_btn_label = "🐙 현재 상태 GitHub(data.json)에 커밋하기" if storage.mode == "github" else "☁️ 현재 상태 클라우드에 저장하기"
        if st.button(save_btn_label, type="primary", use_container_width=True, key="tab5_cloud_save"):
            with st.spinner("클라우드 저장소에 커밋/동기화 중..."):
                ok, msg = storage.save_data(data)
            if ok:
                target_str = f"GitHub({storage.github.repo})" if storage.mode == "github" else "클라우드"
                st.toast(f"✅ {target_str}에 저장되었습니다!")
                st.success(f"✅ {target_str} 저장/커밋 완료! ({get_now().strftime('%Y-%m-%d %H:%M:%S')})")
            else:
                st.error(f"❌ 저장 실패: {msg}")

    with c_save_excel:
        if st.button("📥 엑셀(XLSX) 파일로 다운로드/저장", use_container_width=True, key="tab5_excel_save"):
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

# ==========================================
# TAB 6: 🥗 이유식 영양 & 궁합 코칭 가이드
# ==========================================
with tab6:
    st.subheader("👶 이유식 재료 영양 & 황금 궁합 가이드")
    st.markdown("""
    > 💡 **참고 안내:**  
    > 본 내용은 "절대 금지 음식"이 아니라, 아기의 **소화 편의성, 가스 유발 방지, 철분 흡수 극대화, 맛의 조화**를 고려해 더 균형 있고 편안하게 구성하기 위한 영양 관리 기준입니다.
    """)

    c_g1, c_g2 = st.columns(2)
    with c_g1:
        st.markdown("""
        ### 👑 최고 궁합 라인 (골든 조합)
        *식단 구성 시 가장 추천하는 검증된 황금 조합입니다.*
        - 🥩 **소고기 + 브로콜리 + 감자**: 비타민C가 소고기의 철분(비헴철) 흡수를 극대화
        - 🥩 **소고기 + 브로콜리 + 애호박**: 철분 흡수 촉진 & 소화 편안함 최고
        - 🥩 **소고기 + 애호박 + 당근**: 균형 잡힌 영양 & 편안한 소화
        - 🥩 **소고기 + 감자 + 애호박**: 부드러운 전분질과 담백한 소화
        - 🍗 **닭고기 + 애호박 + 당근**: 닭고기의 담백함과 순한 채소의 환상 궁합
        - 🍗 **닭고기 + 감자 + 양파**: 부드러운 스튜형 최고 궁합
        - 🐟 **흰살생선 + 감자 + 애호박**: 비린내 없이 담백하고 소화 잘 되는 조합
        - 🥕 **애호박 + 감자 + 당근**: 실패 없는 순한 채소 삼총사
        - 🧅 **애호박 + 감자 + 양파**: 풍미와 부드러움을 모두 잡은 채소 조합
        - 🧅 **애호박 + 당근 + 양파**: 달큰하고 향긋한 순한 채소 황금 조합
        - 🥔 **감자 + 당근 + 양파**: 아기들이 가장 좋아하는 기본 영양 채소 조합
        - 🥦 **당근 + 애호박 + 브로콜리**: 비타민과 식이섬유가 균형 잡힌 채소 조합
        - 🥦 **감자 + 애호박 + 브로콜리**: 부드러운 질감과 비타민C 충전 조합
        """)

        st.markdown("""
        ### 🌱 순한 채소 라인
        *소화가 잘 되고 자극이 적어 어느 식단에나 곁들이기 좋은 채소:*
        - **애호박, 감자, 당근, 양파 (소량), 단호박 (소량)**
        """)

    with c_g2:
        st.markdown("""
        ### ⚠️ 주의 & 비추천 조합 라인
        *영양소 파괴, 흡수 방해, 가스 유발 가능성이 있어 피하는 것이 좋은 조합:*
        
        **1. 🥦 십자화과 채소 라인 (가스 가능성 있어 한 끼 1종만 권장)**
        - 재료: **브로콜리, 양배추, 배추, 콜리플라워, 케일, 적채**
        - 🚫 피할 조합: 브로콜리+양배추 / 브로콜리+배추 / 양배추+배추 / 브로콜리+콜리플라워
        
        **2. 🌿 질산염 채소 라인 (한 끼 몰아넣지 않기 / 조리 후 바로 냉동)**
        - 재료: **청경채, 배추, 시금치, 비트, 근대, 상추류**
        - 원칙: 한 끼에 2종 이상 겹치지 않게 분산 급여
        
        **3. 🚫 굳이 안 섞는 비추천 채소 조합**
        - **단호박 + 무**: 비타민C 분해 효소로 인한 영양 파괴
        - **오이 + 당근**: 오이의 아스코르비나아제가 당근 비타민C 산화
        - **감자 + 고구마**: 전분질/탄수화물 과다로 소화 및 배에 가스 유발
        - **애호박 + 단호박**: 유사 호박류 중복
        - **시금치 + 두부/멸치/치즈**: 옥살산과 칼슘이 결합하여 흡수 방해
        - **근대 + 두부**: 옥살산-칼슘 흡수 방해
        """)

        st.markdown("""
        ### 💊 주요 영양소별 채소 정리
        - **베타카로틴 (면역/눈):** 당근, 단호박, 브로콜리
        - **비타민 C (철분 흡수 촉진):** 브로콜리, 감자, 양배추, 배추
        - **엽산 (성장/발달):** 브로콜리, 시금치, 케일
        - **칼륨 (나트륨 배출):** 감자, 단호박, 아보카도
        """)

    st.info("📌 **안내**: 아기의 알레르기 유무, 소화 및 배변 상태에 따라 반응은 달라질 수 있습니다. 특이 반응이 있는 경우 반드시 소아청소년과 전문의와 상담하세요.")
