FROM gcr.io/tensorflow/tensorflow:latest-gpu

RUN apt-get update; \
    apt-get install -y \
    python python-pip \
    python-numpy python-scipy \
    build-essential python-dev python-setuptools \
    libatlas-dev libatlas3gf-base \
    emacs24 python-skimage \
    libhdf5-dev 

RUN update-alternatives --set libblas.so.3 \
    /usr/lib/atlas-base/atlas/libblas.so.3; \
    update-alternatives --set liblapack.so.3 \
    /usr/lib/atlas-base/atlas/liblapack.so.3

RUN pip install -U scikit-learn
RUN pip install -U keras
RUN pip install -U h5py
RUN pip install -U flask

VOLUME /humpbackModel/
RUN mkdir -p /root/.keras/
COPY ./keras.json /root/.keras/

EXPOSE 8888:8888
ENTRYPOINT /bin/bash