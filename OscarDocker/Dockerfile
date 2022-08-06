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
        nano \
        unzip

RUN pip install --upgrade pip
RUN pip install onnx tf2onnx onnx-tf sclblonnx onnxruntime #onnxruntime-gpu
RUN pip install --quiet -U git+https://github.com/microsoft/onnxconverter-common
RUN pip install --quiet -U git+https://github.com/onnx/keras-onnx
RUN pip install seaborn==0.9.0 onnx2keras==0.0.24 skl2onnx==1.10.3
RUN pip install gdown -U --no-cache-dir #Important: resolves the problem of downloading big many files from drive

#Download the model and the script from google drive (DownloadONNXSplitter2 folder)
RUN cd /opt && gdown --folder --no-cookies https://drive.google.com/drive/folders/1YVF-Sxz-f89SP2S77mZD8dWNmeCcxN8d

#LENET
#RUN cd /opt/DownloadONNXSplitter2 && gdown --folder https://drive.google.com/drive/folders/1XgF9MDUVnQS6Xax5tMnTBlZnILyzTwA3
#RUN cd /opt/DownloadONNXSplitter2 && gdown --folder https://drive.google.com/drive/folders/1nV27thljPOBeWs4xgmm0-OJwt36YFtFn

#MobileNetV2
##RUN cd /opt/DownloadONNXSplitter2 && gdown --folder --remaining-ok --no-cookies https://drive.google.com/drive/folders/1DbnLbvsVoxeHZOgX6jgYpPqoLp8iF2vm
RUN cd /opt/DownloadONNXSplitter2 && gdown --id 177YxmYcOjFrceI9GiAVAbvgGH9LRwi6s           #download MobileNetV2_SplittedModels.zip
RUN cd /opt/DownloadONNXSplitter2 && unzip MobileNetV2_SplittedModels.zip
RUN cd /opt/DownloadONNXSplitter2 && rm MobileNetV2_SplittedModels.zip
RUN cd /opt/DownloadONNXSplitter2 && cp mobilenet_v2.onnx MobileNetV2_SplittedModels
RUN cd /opt/DownloadONNXSplitter2 && gdown --folder https://drive.google.com/drive/folders/1yFJA4q7Q8764qSZaIjSLOGnmLN9s70UR

#BlurryFaces
##RUN cd /opt/DownloadONNXSplitter2 && gdown --folder https://drive.google.com/drive/folders/1kYU7gK0EwCu5be3jwsmjP5h0LSk4DvPO
RUN cd /opt/DownloadONNXSplitter2 && gdown --id 1pfOZkVh9yohhTr6rdnx4sRMPirgdUsbi           #download BlurryFaces_SplittedModels.zip 
RUN cd /opt/DownloadONNXSplitter2 && unzip BlurryFaces_SplittedModels.zip
RUN cd /opt/DownloadONNXSplitter2 && rm BlurryFaces_SplittedModels.zip
RUN cd /opt/DownloadONNXSplitter2 && cp blurry_faces.onnx BlurryFaces_SplittedModels

#VGG16
##RUN cd /opt/DownloadONNXSplitter2 && gdown --folder https://drive.google.com/drive/folders/15NrDJUxEddk2QhduJzY7WIozWo5TctvG
#RUN cd /opt/DownloadONNXSplitter2 && gdown --id 1nZrMA7k_IPx0ebkUxN5agxi-5FLSB-i3           #download VGG16_SplittedModels.zip  
#RUN cd /opt/DownloadONNXSplitter2 && unzip VGG16_SplittedModels.zip
#RUN cd /opt/DownloadONNXSplitter2 && rm VGG16_SplittedModels.zip
#RUN cd /opt/DownloadONNXSplitter2 && cp vgg16.onnx VGG16_SplittedModels
