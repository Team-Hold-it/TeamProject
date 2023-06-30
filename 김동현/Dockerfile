FROM python:3.8

RUN apt-get update && \
    apt-get install -y ffmpeg

WORKDIR /app

COPY requirements.txt /app

RUN pip install -r requirements.txt
RUN mkdir -p /app/web/static/downloads
RUN chmod -R 777 /app/web/static/downloads
RUN mkdir -p /app/web/static/images/img
RUN chmod -R 777 /app/web/static/images/img
RUN mkdir -p /app/web/static/images/predict
RUN chmod -R 777 /app/web/static/images/predict
RUN mkdir -p /app/web/runs/detect/predict/
RUN chmod -R 777 /app/web/runs/detect/predict/
RUN mkdir -p /app/web/static/capture/
RUN chmod -R 777 /app/web/static/capture/

COPY . /app

EXPOSE 8000

CMD ["python", "web.py"]
