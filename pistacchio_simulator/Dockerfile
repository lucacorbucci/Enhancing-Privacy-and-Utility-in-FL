FROM nvcr.io/nvidia/pytorch:22.11-py3
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get -y install sudo
# RUN sudo apt-get -y install python3.8
# RUN python -m ensurepip --upgrade
RUN useradd -u 1005 -m lucacorbucci
# COPY ./src/requirements.txt . 
# RUN pip install -r requirements.txt
