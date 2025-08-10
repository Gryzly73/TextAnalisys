# 🧠 Анализатор клиентских отзывов

Веб-приложение для анализа тональности и извлечения ключевых слов из клиентских отзывов с использованием NLP.

![Demo Screenshot](demo.png)

## ✨ Возможности

- Анализ тональности отзывов (положительный/нейтральный/отрицательный)
- Визуализация распределения тональности
- Генерация облака ключевых слов
- Поддержка русского и английского языков 
- Загрузка собственных данных (CSV/Excel)
- Экспорт результатов в CSV

## ⚙️ Технологии

- **NLP**: Hugging Face Transformers
- **Визуализация**: Matplotlib, WordCloud
- **Обработка данных**: Pandas
- **Интерфейс**: Streamlit
- **Модели**:
  - Русский: `blanchefort/rubert-base-cased-sentiment`
  - Английский: Стандартная модель sentiment-analysis

## 🚀 Запуск приложения

1. Клонируйте репозиторий:
```bash
git clone https://github.com/ваш-username/text-analysis-app.git
cd text-analysis-app

2. Установите зависимости:
pip install -r requirements.txt

3. Запустите приложение:
streamlit run app.py

4. Откройте в браузере: http://localhost:8501
