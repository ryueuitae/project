# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import implicit
from scipy.sparse import csr_matrix
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import logging
import os

# 한글 폰트 설정 (로컬/클라우드 모두 대응)
def set_korean_font():
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
    except:
        try:
            fe = fm.FontEntry(fname='/usr/share/fonts/truetype/nanum/NanumGothic.ttf', name='NanumGothic')
            fm.fontManager.ttflist.insert(0, fe)
            plt.rcParams['font.family'] = 'NanumGothic'
            plt.rcParams['axes.unicode_minus'] = False
        except:
            plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

logging.basicConfig(level=logging.INFO)

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except:
    BASE_DIR = os.getcwd()

DATA_PATH = BASE_DIR

ERA_MAP = {
    '50': '1950s', '60': '1960s', '70': '1970s', '80': '1980s',
    '90': '1990s', '00': '2000s', '10': '2010s', '19': '2019'
}
ERA_ORDER = ['1950s','1960s','1970s','1980s','1990s','2000s','2010s','2019']

@st.cache_data
def load_data():
    try:
        rate_df = pd.read_csv(os.path.join(DATA_PATH, 'rate_data.csv'), encoding='utf-8-sig')
        item_df = pd.read_csv(os.path.join(DATA_PATH, 'item_data.csv'), encoding='utf-8-sig')
        return rate_df, item_df
    except FileNotFoundError as e:
        st.error(f"데이터 파일을 찾을 수 없습니다: {e}")
        st.stop()

@st.cache_resource
def train_model(rate_df):
    try:
        user_cat = rate_df['user'].astype('category')
        item_cat = rate_df['item'].astype('category')
        user_ids = user_cat.cat.codes
        item_ids = item_cat.cat.codes
        sparse = csr_matrix(
            (rate_df['rate'], (user_ids, item_ids)),
            shape=(user_ids.max() + 1, item_ids.max() + 1)
        )
        model = implicit.als.AlternatingLeastSquares(
            factors=50, iterations=30, random_state=42, use_gpu=False
        )
        model.fit(sparse)
        return model, sparse, user_cat.cat.categories, item_cat.cat.categories
    except Exception as e:
        st.error(f"모델 학습 오류: {e}")
        st.stop()

@st.cache_data
def get_als_popular_items(_model, _sparse, _item_categories, sample_users=200):
    n_users = _sparse.shape[0]
    sample  = min(sample_users, n_users)
    counter = Counter()
    for user_code in range(sample):
        user_items = _sparse[user_code]
        ids, scores = _model.recommend(
            user_code, user_items, N=20, filter_already_liked_items=True
        )
        for item_code, score in zip(ids, scores):
            counter[item_code] += score
    rec_df = pd.DataFrame(counter.items(), columns=['item_code', 'als_score'])
    rec_df['item'] = rec_df['item_code'].apply(
        lambda x: _item_categories[x] if x < len(_item_categories) else None
    )
    return rec_df.dropna().sort_values('als_score', ascending=False)

def extract_era(item_name):
    try:
        return ERA_MAP.get(item_name.split('_')[2], 'etc')
    except:
        return 'etc'

@st.cache_data
def get_trend_data(rate_df, item_df):
    merged = rate_df.merge(item_df[['item', 'item_name', 'style', 'gender']], on='item')
    merged['era'] = merged['item_name'].apply(extract_era)
    return merged

# --- 앱 시작 ---
st.set_page_config(page_title="패션 사입 추천 시스템", page_icon="👗", layout="wide")
st.title("👗 패션 사입 추천 시스템")
st.markdown("AI 기반으로 **지금 뜨는 스타일**과 **사입할 아이템**을 찾아드립니다 ✨")

with st.spinner("데이터 및 AI 모델 로딩 중..."):
    rate_df, item_df                  = load_data()
    model, sparse, user_cat, item_cat = train_model(rate_df)
    trend_df                          = get_trend_data(rate_df, item_df)

with st.spinner("AI가 트렌드 아이템 분석 중..."):
    als_popular = get_als_popular_items(model, sparse, item_cat)

tab1, tab2 = st.tabs(["📊 트렌드 분석", "🛍️ AI 사입 추천"])

# =====================
# 탭1: 트렌드 분석
# =====================
with tab1:
    st.subheader("📈 연도별 스타일 트렌드")
    col1, col2 = st.columns(2)

    with col1:
        era_trend = trend_df.groupby('era')['rate'].mean().reindex(ERA_ORDER).dropna()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(len(era_trend)), era_trend.values, marker='o',
                color='tomato', linewidth=2.5, markersize=8)
        ax.fill_between(range(len(era_trend)), era_trend.values, alpha=0.15, color='tomato')
        ax.set_xticks(range(len(era_trend)))
        ax.set_xticklabels(era_trend.index, rotation=45, ha='right', fontsize=9)
        ax.set_title('Era Trend (Avg Rating)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Avg Rating')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        style_rank = trend_df.groupby('style')['rate'].mean().sort_values(ascending=False).head(10)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        colors = ['#e63946' if i == 0 else '#457b9d' for i in range(len(style_rank))]
        ax2.barh(style_rank.index[::-1], style_rank.values[::-1], color=colors[::-1])
        ax2.set_title('Top 10 Popular Styles', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Avg Rating')
        for i, val in enumerate(style_rank.values[::-1]):
            ax2.text(val + 0.01, i, f'{val:.2f}', va='center', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("🔥 Rising Styles (버터떡 찾기)")

    recent_eras  = ['2010s', '2019', '1990s']
    past_eras    = ['1950s', '1960s', '1970s', '1980s']
    recent_score = trend_df[trend_df['era'].isin(recent_eras)].groupby('style')['rate'].mean()
    past_score   = trend_df[trend_df['era'].isin(past_eras)].groupby('style')['rate'].mean()
    combined     = pd.DataFrame({'recent': recent_score, 'past': past_score}).fillna(0)
    rising       = (combined['recent'] - combined['past']).sort_values(ascending=False)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**📈 상승 중인 스타일 Top 5**")
        top_rising = rising.head(5)
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        ax3.bar(top_rising.index, top_rising.values, color='#2dc653')
        ax3.set_title('Rising Styles', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Score Change')
        plt.xticks(rotation=30, ha='right', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig3)

    with col4:
        st.markdown("**📉 하락 중인 스타일 Top 5**")
        top_falling = rising.tail(5)
        fig4, ax4 = plt.subplots(figsize=(6, 3))
        ax4.bar(top_falling.index, top_falling.values, color='#e63946')
        ax4.set_title('Falling Styles', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Score Change')
        plt.xticks(rotation=30, ha='right', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig4)

# =====================
# 탭2: AI 사입 추천
# =====================
with tab2:
    st.subheader("🤖 AI 사입 추천")
    st.markdown("ALS 모델이 **200명의 유저 패턴**을 학습해서 가장 많이 추천된 아이템을 뽑았어요")

    st.sidebar.header("🔍 조건 입력")
    gender_opts     = ["전체"] + sorted(item_df['gender'].unique().tolist())
    season_opts     = ["전체", "spring", "summer", "fall", "winter"]
    style_opts      = ["전체"] + sorted(item_df['style'].unique().tolist())
    selected_gender = st.sidebar.selectbox("성별", gender_opts)
    selected_season = st.sidebar.selectbox("시즌", season_opts)
    selected_style  = st.sidebar.selectbox("스타일", style_opts)
    top_n           = st.sidebar.slider("추천 개수", 5, 20, 10)

    result = als_popular.merge(
        item_df[['item', 'item_name', 'style', 'gender', 'season', 'tpo']], on='item'
    )

    if selected_gender != "전체":
        result = result[result['gender'] == selected_gender]
    if selected_season != "전체":
        result = result[result['season'].str.contains(selected_season, na=False)]
    if selected_style != "전체":
        result = result[result['style'] == selected_style]

    result = result.head(top_n).reset_index(drop=True)
    result.index += 1

    if result.empty:
        st.warning("조건에 맞는 아이템이 없어요! 조건을 바꿔보세요.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("추천 아이템 수", len(result))
        m2.metric("TOP 스타일", result['style'].iloc[0])
        m3.metric("AI 추천 점수 (1위)", f"{result['als_score'].iloc[0]:.1f}")

        st.markdown("---")

        display = result[['item_name', 'style', 'gender', 'season', 'als_score']].copy()
        display.columns = ['아이템명', '스타일', '성별', '시즌', 'AI 추천점수']
        display['AI 추천점수'] = display['AI 추천점수'].round(2)
        st.dataframe(display, use_container_width=True)

        st.markdown("#### 추천 아이템 스타일 분포")
        style_dist = result['style'].value_counts()
        fig5, ax5 = plt.subplots(figsize=(8, 3))
        ax5.bar(style_dist.index, style_dist.values, color='#457b9d')
        ax5.set_title('Style Distribution of Recommended Items', fontweight='bold')
        ax5.set_ylabel('Count')
        plt.xticks(rotation=30, ha='right', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig5)

st.markdown("---")
st.caption("AI Hub 연도별 패션 선호도 데이터 기반 | 사회과학기반 디자인씽킹 BM")