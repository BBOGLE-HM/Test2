"""AI 관상가 - Streamlit + OpenAI Vision 관상 풀이 앱"""
import base64
import io
import os

import streamlit as st
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
from PIL import Image, ImageOps, UnidentifiedImageError

load_dotenv()

MAX_FILE_MB = 5
MAX_SIDE = 1024
MAX_CALLS_PER_SESSION = 10
FOCUS_OPTIONS = ["종합", "재물", "연애", "직업", "건강"]
TONE_OPTIONS = {"친근한 말투": "친근한 말투", "전통 관상가 말투": "전통 관상가 말투(~하오, ~이로다 풍)"}

SYSTEM_PROMPT = """당신은 30년 경력의 친근한 관상가입니다. 사용자가 올린 얼굴 사진을 보고 오락 목적의 관상 풀이를 해주세요.

[원칙]
- 한국어로, 따뜻하고 긍정적이며 건설적인 톤으로 답합니다.
- 사진에서 실제로 보이는 특징(이마, 눈, 코, 입, 귀, 턱/얼굴형 등)을 근거로 풀이합니다.
- 외모 비하, 인종·성별·나이·장애에 대한 차별적 표현, 질병 진단이나 단정적인 예언은 하지 않습니다.
- 부정적인 해석이 나오는 부분은 "이렇게 보완하면 좋아요"처럼 조언 형태로 바꿔 말합니다.
- 사진 속 인물의 신원, 실명, 민감한 개인정보를 추측하지 않습니다.
- 사진에 사람 얼굴이 전혀 보이지 않을 때(풍경, 사물, 동물 등)에만 풀이 대신 "얼굴이 잘 보이는 정면 사진을 다시 올려주세요."라고만 답합니다. 약간 비스듬하거나, 어둡거나, 안경·모자·마스크가 있어도 얼굴이 보이면 보이는 특징만으로 풀이하세요.
- 이 풀이는 오락용 성격 풀이이므로 사진 속 인물을 식별하려는 것이 아닙니다. 보이는 외형 특징만 이야기하면 됩니다.
- 얼굴이 여러 명이면 가장 크게 보이는 한 명만 풀이합니다.
- 마지막에 "이 풀이는 재미로 봐주세요." 문구를 한 줄 덧붙입니다.

[출력 형식 - 마크다운]
### ✨ 첫인상 한 줄 요약
(한 문장)

### 🔍 부위별 풀이
- **이마 (초년운)**: ...
- **눈 (성격·지혜)**: ...
- **코 (재물운)**: ...
- **입 (말·식복)**: ...
- **귀 (복·장수)**: ...
- **턱·얼굴형 (말년운)**: ...

### 🧭 종합 성격
(3~4문장)

### 🍀 운세 키워드
| 구분 | 점수 | 한줄 코멘트 |
|---|---|---|
| 재물운 | ★★★★☆ | ... |
| 연애운 | ★★★☆☆ | ... |
| 직업운 | ★★★★★ | ... |
| 건강운 | ★★★★☆ | ... |

### 💬 한마디 조언
(1~2문장)"""

USER_PROMPT = """첨부한 얼굴 사진으로 관상을 봐주세요.
- 중점 분야: {focus}
- 말투: {tone}"""

NO_FACE_MARK = "얼굴이 잘 보이는 정면 사진을 다시 올려주세요"


def get_secret(name: str) -> str | None:
    """st.secrets -> 환경변수(.env) 순으로 조회."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name)


def prepare_image(file_bytes: bytes) -> str:
    """이미지를 RGB·리사이즈 후 base64 data URL로 변환."""
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def analyze(data_url: str, focus: str, tone: str) -> str:
    client = OpenAI(api_key=get_secret("OPENAI_API_KEY"), timeout=60)
    model = get_secret("OPENAI_MODEL") or "gpt-4o-mini"
    resp = client.chat.completions.create(
        model=model,
        temperature=0.8,
        max_tokens=1200,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT.format(focus=focus, tone=TONE_OPTIONS[tone])},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ],
            },
        ],
    )
    return resp.choices[0].message.content or ""


CSS = """
<style>
.stApp { background: linear-gradient(180deg, #1b1230 0%, #2b1b4a 45%, #3a2560 100%); }
.block-container { max-width: 720px; padding-top: 2rem; }
.hero { text-align: center; padding: 1.2rem 0 0.6rem; }
.hero h1 { font-size: 2.4rem; margin: 0; color: #f6e7b4; letter-spacing: 2px; }
.hero p { color: #d9cdf2; margin-top: .4rem; }
.notice { background: rgba(255,255,255,.08); border: 1px solid rgba(246,231,180,.25);
          border-radius: 12px; padding: .7rem 1rem; color: #e8def8; font-size: .85rem; margin-bottom: 1rem; }
label, .stSelectbox label, .stFileUploader label { color: #f0e8ff !important; }
div.stButton > button[kind="primary"] { width: 100%; background: #d4af37; color: #2a1f3d;
               font-weight: 700; border: none; border-radius: 12px; padding: .7rem; }
div.stButton > button[kind="primary"]:hover { background: #e6c35a; color: #2a1f3d; }
.footer { text-align: center; color: #b9aad6; font-size: .78rem; margin-top: 2rem; }
@media (max-width: 600px) { .hero h1 { font-size: 1.8rem; } }
</style>
"""


def main() -> None:
    st.set_page_config(page_title="AI 관상가", page_icon="🔮", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><h1>🔮 AI 관상가</h1>'
        "<p>얼굴 사진을 올리면 AI가 관상을 풀이해 드려요.<br>재미로만 봐주세요!</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="notice">🔒 이미지는 저장되지 않으며 분석을 위해 OpenAI로 전송됩니다.<br>'
        "👤 본인 또는 동의받은 사진만 사용하세요.</div>",
        unsafe_allow_html=True,
    )

    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY가 설정되지 않았습니다. .env 또는 Streamlit Secrets를 확인하세요.")

    uploaded = st.file_uploader("얼굴 사진 업로드", type=["jpg", "jpeg", "png"])
    file_bytes = None
    if uploaded:
        file_bytes = uploaded.getvalue()
        if len(file_bytes) > MAX_FILE_MB * 1024 * 1024:
            st.warning(f"파일이 {MAX_FILE_MB}MB를 넘습니다. 더 작은 사진을 올려주세요.")
            file_bytes = None
        else:
            st.image(file_bytes, width="stretch")

    col1, col2 = st.columns(2)
    focus = col1.selectbox("중점 분야", FOCUS_OPTIONS)
    tone = col2.selectbox("풀이 말투", list(TONE_OPTIONS))

    calls = st.session_state.setdefault("calls", 0)
    limit_reached = calls >= MAX_CALLS_PER_SESSION
    if limit_reached:
        st.warning("이번 접속에서 사용할 수 있는 횟수를 모두 사용했어요. 새로고침 후 다시 이용해주세요.")

    if st.button("관상 보기", type="primary", disabled=not (file_bytes and api_key) or limit_reached):
        try:
            data_url = prepare_image(file_bytes)
        except (UnidentifiedImageError, OSError):
            st.error("이미지를 읽을 수 없습니다. 다른 파일을 올려주세요.")
            return
        try:
            with st.spinner("관상을 보는 중이에요... 🔮"):
                result = analyze(data_url, focus, tone)
        except AuthenticationError:
            st.error("API 키가 올바르지 않습니다.")
        except RateLimitError:
            st.error("요청이 너무 많거나 사용 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
        except (APITimeoutError, APIConnectionError):
            st.error("OpenAI 서버와 연결이 원활하지 않습니다. 다시 시도해주세요.")
        except OpenAIError as e:
            st.error(f"분석 중 오류가 발생했습니다: {e.__class__.__name__}")
        else:
            if not result.strip():
                st.error("결과를 받지 못했어요. 다시 시도해주세요.")
            elif NO_FACE_MARK in result and len(result) < 120:
                st.warning(result)
            else:
                st.session_state["calls"] = calls + 1
                st.session_state["result"] = result
                st.session_state["result_key"] = hash(file_bytes)

    # 결과는 세션에 보관 (다른 위젯 조작 시 사라지지 않게). 같은 사진일 때만 표시
    if file_bytes and st.session_state.get("result_key") == hash(file_bytes):
        with st.container(border=False):
            st.markdown("#### 📜 관상 풀이 결과")
            st.markdown(st.session_state["result"])
        st.download_button(
            "결과 텍스트 저장", st.session_state["result"], file_name="gwansang.md", mime="text/markdown"
        )

    st.markdown(
        '<div class="footer">※ 본 서비스는 오락 목적이며 실제 판단의 근거가 될 수 없습니다.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
