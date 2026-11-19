# FootPredict as a plain container: the same Streamlit app, served directly (no hosting wrapper page, so no
# "Hosted with Streamlit" badge on a full page load). Data and models come from the repository (local_data/), which
# the daily GitHub workflow updates; a host with auto-deploy on push picks every update up.
#
#   docker build -t footpredict . && docker run -p 8501:8501 footpredict
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN useradd --create-home footpredict && chown -R footpredict /app
USER footpredict

EXPOSE 8501
HEALTHCHECK --interval=60s --timeout=10s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8501\")}/_stcore/health')"
# hosts such as Render pass the port in $PORT
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false"]
