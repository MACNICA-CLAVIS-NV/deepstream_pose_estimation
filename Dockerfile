ARG BASE_IMAGE=nvcr.io/nvidia/deepstream-l4t:5.1-21.02-samples
FROM ${BASE_IMAGE}

ARG REPOSITORY_NAME=deepstream_pose_estimation

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

WORKDIR /tmp

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libjson-glib-dev \
        graphviz && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /${REPOSITORY_NAME}
COPY ./ /${REPOSITORY_NAME}

WORKDIR /${REPOSITORY_NAME}

RUN make

RUN /usr/src/tensorrt/bin/trtexec --onnx=pose_estimation.onnx --saveEngine=pose_estimation.onnx_b1_gpu0_fp16.engine

CMD ./deepstream-pose-estimation-app /dev/video0