FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[mlflow]"

# Generate the synthetic CRM history, train, and export the Odoo demo at build time (about a minute).
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db MLFLOW_DISABLE_AGENT_HINT=1
RUN python -m crm_score_ml.generator --out data/generated/v1 \
    && python -m crm_score_ml.train --data data/generated/v1 --out models/latest \
    && python -m crm_score_ml.export_demo --data data/generated/v1 --model models/latest --out /app/export/demo.json

ENV MODEL_DIR=/app/models/latest
EXPOSE 8000
CMD ["uvicorn", "crm_score_ml.service:app", "--host", "0.0.0.0", "--port", "8000"]
