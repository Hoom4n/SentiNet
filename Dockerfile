FROM python:3.11-slim

WORKDIR /app

COPY ./app ./app
COPY ./saved_models ./saved_models
COPY ./requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
   gcc \
    python3-dev \
   && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 7861

CMD ["python", "app/app.py"]
