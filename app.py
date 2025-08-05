# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer

# Конфигурация страницы
st.set_page_config(
    page_title="Анализ отзывов",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Анализатор клиентских отзывов")
st.caption("Быстрое получение инсайтов из текстов с помощью NLP")

# --- Загрузка примера данных --- #
@st.cache_data
def load_sample_data(lang='ru'):
    if lang == 'ru':
        data = {
            'text': [
                "Отличный товар, быстрая доставка!",
                "Качество хорошее, но доставка задержалась",
                "Не соответствует описанию, очень разочарован",
                "Хороший магазин, рекомендую",
                "Ужасное обслуживание, не советую"
            ],
            'rating': [5, 4, 2, 5, 1]
        }
    else:
        data = {
            'text': [
                "Great product, fast delivery!",
                "Good quality but shipping was late",
                "Very disappointed, not as described",
                "Nice shop, would recommend",
                "Terrible service, not satisfied"
            ],
            'rating': [5, 4, 2, 5, 1]
        }
    return pd.DataFrame(data)

# --- Источник данных --- #
data_source = st.radio("📥 Выберите источник данных:", ["Пример данных", "Загрузить файл"])

if data_source == "Пример данных":
    df = load_sample_data('ru')
    reviews = df['text'].tolist()
    st.success(f"Загружено {len(reviews)} примерных отзывов")
    st.dataframe(df.head())
else:
    uploaded_file = st.file_uploader("Загрузите файл (CSV или Excel)", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Поиск колонки с текстом
            text_cols = [col for col in df.columns if 'text' in col.lower() or 'review' in col.lower()]
            if text_cols:
                reviews = df[text_cols[0]].dropna().astype(str).tolist()
                st.success(f"Загружено {len(reviews)} отзывов из файла")
            else:
                st.error("Не найдена колонка с отзывами (например, 'text' или 'review')")
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")
    else:
        reviews = []

# --- Анализ --- #
if reviews:
    with st.expander("⚙️ Настройки анализа"):
        language = st.radio("Язык отзывов", ["Русский", "Английский"], index=0)
        max_reviews = st.slider("Сколько отзывов анализировать", 10, min(1000, len(reviews)), 100)
        show_details = st.checkbox("Показать таблицу с результатами", value=True)

    reviews = reviews[:max_reviews]

    @st.cache_resource
    def load_models(lang):
        if lang == "Русский":
            sentiment = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")
        else:
            sentiment = pipeline("sentiment-analysis")

        vectorizer = CountVectorizer(
            max_features=50,
            stop_words=["это", "и", "в"] if lang == "Русский" else ["the", "and", "in"]
        )
        return sentiment, vectorizer

    sentiment_model, vectorizer = load_models(language)

    if st.button("🔍 Запустить анализ", type="primary"):
        with st.spinner("Обработка отзывов..."):
            sentiments = sentiment_model(reviews)
            X = vectorizer.fit_transform(reviews)
            keywords = vectorizer.get_feature_names_out()

            results = []
            for i, (text, sent) in enumerate(zip(reviews, sentiments)):
                words = [w for w in text.lower().split() if w in keywords]
                results.append({
                    "Отзыв": text,
                    "Тональность": sent['label'],
                    "Уверенность": round(sent['score'], 3),
                    "Ключевые слова": ", ".join(set(words))
                })

            df_results = pd.DataFrame(results)

        st.subheader("📊 Результаты анализа")

        # График по тональности
        fig1, ax1 = plt.subplots()
        df_results['Тональность'].value_counts().plot(
            kind='bar',
            color=['lightgreen', 'salmon', 'gray'],
            ax=ax1
        )
        ax1.set_title("Распределение тональности")
        st.pyplot(fig1)

        # Облако слов
        st.subheader("☁️ Облако ключевых слов")
        wordcloud = WordCloud(width=800, height=400).generate(" ".join(reviews))
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)

        # Таблица
        if show_details:
            st.subheader("📋 Детализация по каждому отзыву")
            st.dataframe(df_results, use_container_width=True)

        # Кнопка скачивания
        csv = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Скачать CSV",
            data=csv,
            file_name="results.csv",
            mime="text/csv"
        )

# --- Инструкция --- #
with st.expander("ℹ️ Инструкция по использованию"):
    st.markdown("""
    1. Выберите источник данных — пример или загрузите свой файл (CSV/Excel)
    2. Убедитесь, что колонка с отзывами содержит текст
    3. Нажмите кнопку **Запустить анализ**
    4. Ознакомьтесь с графиками и таблицей
    5. Скачайте CSV с результатами
    """)
