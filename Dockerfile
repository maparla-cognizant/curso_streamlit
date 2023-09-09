FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY qa.py qa.py

RUN pip3 install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "qa.py", "--server.port=8501", "--server.address=0.0.0.0"]