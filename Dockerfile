FROM ubuntu:20.04

SHELL ["/bin/bash", "-exo", "pipefail", "-c"]

RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker \
    && echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker

RUN DEBIAN_FRONTEND=noninteractive \
  apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y python3 pip python3-tk libgl1-mesa-glx ffmpeg libsm6 libxext6 libjpeg-dev libpng-dev \
  && rm -rf /var/lib/apt/lists/* \
  && mkdir -p /root/toolkit

COPY ./cu_seqlens.npy /root/toolkit

COPY ./rotary_pos_emb.npy /root/toolkit

COPY ./export_vision.py /root/toolkit

COPY ./rkllm_toolkit-1.1.4-cp38-cp38-linux_x86_64.whl /root/toolkit

ADD ./data/ /root/toolkit/data

RUN python3 -m pip install rknn-toolkit2==2.2.1

RUN python3 -m pip install /root/toolkit/rkllm_toolkit-1.1.4-cp38-cp38-linux_x86_64.whl

RUN python3 -m pip install --upgrade torch torchvision pillow inquirer

WORKDIR /root/toolkit

CMD /usr/bin/python3 ./export_vision.py