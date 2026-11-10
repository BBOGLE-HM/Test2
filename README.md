# 🔮 AI 관상가

얼굴 사진을 업로드하면 OpenAI Vision 모델이 관상을 풀이해 주는 Streamlit 앱 (오락용).

## 로컬 실행
```bash
pip install -r requirements.txt
cp .env.example .env     # OPENAI_API_KEY 입력
streamlit run app.py
```

## Streamlit Cloud 배포
1. 이 폴더를 GitHub에 push (`.env`는 `.gitignore`로 제외됨)
2. https://share.streamlit.io 에서 저장소·브랜치·`app.py` 선택
3. Advanced settings → Secrets에 입력
   ```toml
   OPENAI_API_KEY = "sk-..."
   OPENAI_MODEL = "gpt-4o-mini"
   ```
