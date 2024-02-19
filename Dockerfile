FROM python:3.11-slim-bullseye
MAINTAINER Alasdair Taylor


WORKDIR /opt/classifier

RUN apt-get update &&\
    apt-get install --no-install-recommends --yes build-essential

RUN pip install --upgrade pip

COPY requirements.txt /opt/classifier/requirements.txt
RUN pip install -r requirements.txt
# COPY . /opt/classifier



