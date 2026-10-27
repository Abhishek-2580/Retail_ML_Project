FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY data ./data

ENV STREAMLIT_THEME_BASE=light \
    STREAMLIT_THEME_PRIMARY_COLOR=#4F46E5 \
    STREAMLIT_THEME_BACKGROUND_COLOR=#F8FAFC \
    STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR=#EEF2FF \
    STREAMLIT_THEME_TEXT_COLOR=#0F172A

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
