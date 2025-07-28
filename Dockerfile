FROM python:3.9-slim

WORKDIR /app

# Copy necessary files
COPY ./app ./app
COPY ./saved_models ./saved_models
COPY ./requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
   gcc \
    python3-dev \
   && rm -rf /var/lib/apt/lists/*


#Requirments
RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 7861

CMD ["python", "app/app.py"]
