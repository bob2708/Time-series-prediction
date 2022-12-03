# syntax=docker/dockerfile:1

FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY testovoe.py testovoe.py
COPY extraction.py extraction.py

CMD [ "python3", "testovoe.py" ]