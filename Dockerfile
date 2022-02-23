FROM python:3.9-slim

COPY . /nmtf
WORKDIR /nmtf/
RUN pip install -U pip && pip install -r requirements.txt && pip install .
WORKDIR /
RUN rm -rf nmtf
