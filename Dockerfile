FROM python:3.6-slim
RUN apt-get update
RUN apt-get install -y git gcc python3-dev
RUN pip3 install setuptools
COPY . /install
RUN cd /install/ || exit 1 && python setup.py install;
RUN rm -rfd /install/ || exit 1;
