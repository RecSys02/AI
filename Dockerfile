FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir uvicorn fastapi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
WORKDIR /app/fastapi_app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
