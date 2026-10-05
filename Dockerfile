FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 user
WORKDIR /app
COPY requirements.txt requirements_streamlit.txt ./
RUN pip install --no-cache-dir -r requirements_streamlit.txt
COPY --chown=user:user config.py streamlit_app.py ./
COPY --chown=user:user src ./src
COPY --chown=user:user assets ./assets
COPY --chown=user:user .streamlit ./.streamlit
COPY --chown=user:user data/documents ./data/documents
RUN mkdir -p /app/db/chroma && chown -R user:user /app
USER user
ENV PYTHONUNBUFFERED=1 XDG_CACHE_HOME=/home/user/.cache
EXPOSE 7860
CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true", "--browser.gatherUsageStats=false"]
