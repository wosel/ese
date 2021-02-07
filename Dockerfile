FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN apt-get update -y && apt-get install ffmpeg libsm6 libxext6  -y
ARG MODELPATH
ADD ${MODELPATH} /src/model.pth

ADD requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
RUN mkdir -p /src
#ADD train.ipynb /src
ADD run.py /src
ADD run.sh /src
ADD dataset.py /src
RUN chmod a+x /src/run.sh

WORKDIR /
ENTRYPOINT ["./src/run.sh"]
