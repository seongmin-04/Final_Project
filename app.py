import streamlit as st       # 웹 UI 라이브러리
import pandas as pd          # 데이터 프레임 처리
import joblib                # 저장된 파일 로드
import matplotlib.pyplot as plt # 그래프 도화지
import seaborn as sns        # 시각화 라이브러리

st.set_page_config(page_title="기대수명 예측 웹 서비스", layout="wide") # 웹페이지 기본 설정
st.title("다중 특성 회귀 모델 기반 기대수명 예측기") # 메인 제목 출력
st.markdown("**(Linear vs Polynomial vs Ridge 예측 결과를 실시간으로 비교하고 테스트합니다.)**") # 부제목 출력

def load_assets():           # 저장된 파일들을 불러오는 함수
    models = {
        'Linear': joblib.load('Linear_model.pkl'), # 선형 모델 로드
        'Poly': joblib.load('Poly_model.pkl'),     # 다항 모델 로드
        'Ridge': joblib.load('Ridge_model.pkl')    # 릿지 모델 로드
    }
    metrics_df = joblib.load('metrics.pkl')        # 성능 지표 표 로드
    return models, metrics_df

models, metrics_df = load_assets() # 함수 실행하여 변수에 담기

st.sidebar.header("입력 파라미터 조절") # 왼쪽 사이드바 제목

adult_mortality = st.sidebar.slider("성인 사망률 (Adult Mortality)", 1, 1000, 150) # 사망률 슬라이더
bmi = st.sidebar.slider("체질량지수 (BMI)", 1.0, 100.0, 38.0)                      # BMI 슬라이더
gdp = st.sidebar.slider("1인당 GDP", 1.0, 150000.0, 5000.0)                      # GDP 슬라이더
alcohol = st.sidebar.slider("알코올 소비량 (Alcohol)", 0.0, 20.0, 5.0)             # 알코올 슬라이더

st.sidebar.markdown("---") # 사이드바 가로줄
selected_model_name = st.sidebar.selectbox("로딩할 예측 모델 선택", ['Linear', 'Poly', 'Ridge']) # 모델 선택 드롭다운

st.subheader("1. 모델 성능 비교 (과대적합 및 규제 효과 관찰)") # 섹션 1 제목

st.markdown("**[성능 평가지표 테이블]**") # 표 소제목
st.dataframe(metrics_df)                  # 성능 지표 표 출력

st.markdown("**[Test 점수 비교 막대그래프]**") # 그래프 소제목
fig, ax = plt.subplots(figsize=(7, 4))         # 그래프 도화지 생성
sns.barplot(x='Model', y='Test R^2', data=metrics_df, ax=ax, palette='viridis') # Test R^2 막대그래프 그리기
st.pyplot(fig)                                 # 완성된 그래프 출력

st.markdown("---") # 메인 화면 가로줄
st.subheader(f"2. 실시간 기대수명 예측 결과 (선택된 모델: {selected_model_name})") # 섹션 2 동적 제목

input_data = pd.DataFrame({ # 사이드바 입력값들을 데이터프레임으로 변환
    'Adult mortality': [adult_mortality],
    'BMI': [bmi],
    'GDP': [gdp],
    'Alcohol': [alcohol]
})

model = models[selected_model_name]       # 드롭다운에서 선택한 모델 꺼내기
prediction = model.predict(input_data)[0] # 사용자가 입력한 데이터로 기대수명 예측

st.markdown( # 예측 결과를 초록색 큰 글씨 박스로 예쁘게 출력
    f"<h1 style='text-align: center; color: #4CAF50; background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>"
    f"예측된 기대수명: {prediction:.2f} 세</h1>",
    unsafe_allow_html=True
)
