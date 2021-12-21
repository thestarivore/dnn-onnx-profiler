FROM python:3.7

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        curl \
        git \
        python-setuptools \
        python3-pip \
        build-essential \
        python-dev \
        python3-wheel \
        python3-numpy \
        python3-scipy \
        python-tk \
        nano

RUN pip install --upgrade pip
RUN pip install onnx tf2onnx onnx-tf sclblonnx
RUN pip install --quiet -U git+https://github.com/microsoft/onnxconverter-common
RUN pip install --quiet -U git+https://github.com/onnx/keras-onnx
RUN pip install seaborn==0.9.0 onnx2keras==0.0.24 skl2onnx==1.10.3
RUN pip install gdown

#Download the model and the script from google drive
RUN cd /opt && gdown --folder https://drive.google.com/drive/folders/1YVF-Sxz-f89SP2S77mZD8dWNmeCcxN8d
RUN cd /opt/DownloadONNXSplitter2 && gdown --folder https://drive.google.com/drive/folders/1XgF9MDUVnQS6Xax5tMnTBlZnILyzTwA3
RUN cd /opt/DownloadONNXSplitter2 && gdown --folder https://drive.google.com/drive/folders/1nV27thljPOBeWs4xgmm0-OJwt36YFtFn

        
