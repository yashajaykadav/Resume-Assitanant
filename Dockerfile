#dockerFile

FROM python:3.11-slim

WORKDIR /app

#Install system dependencies

RUN apt-get update &&  apt-get Install -y \
        gcc \
        g++ \
        && rm -rf  /var/lib/apt/lists/*

#Copy requirements

COPY requirements.txt  .
RUN pip install --no-cache-dir -r requirements.txt

#COPY application code
COPY . .

#Create nesesory directories
RUN mkdir -p upload data/chroma logs

#port
EXPOSE 8000

#RUN APPLICATION
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]