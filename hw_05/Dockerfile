FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

COPY requirements.txt .

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 9696

CMD ["python", "lead_score_conversion_prediction.py"]
