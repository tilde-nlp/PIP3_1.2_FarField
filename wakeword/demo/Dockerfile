FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get -y install python3 python3-pip
RUN pip install numpy==1.22.4

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY . /usr/src/app

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update \
        && apt-get install -y --no-install-recommends dialog

RUN apt-get install -y libsndfile1 wget

RUN cd models && \
    wget https://github.com/harvard-edge/multilingual_kws/releases/download/v0.1-alpha/multilingual_context_73_0.8011.tar.gz && \
    tar -xzf multilingual_context_73_0.8011.tar.gz && \ 
    rm multilingual_context_73_0.8011.tar.gz

EXPOSE 8080 8083

ENTRYPOINT ["python3", "main.py"]
