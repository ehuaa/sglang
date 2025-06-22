ARG CUDA_VERSION=12.4.1

# use 24.02 to cope with tianqing cuda version 12.3.X
FROM nvcr.io/nvidia/tritonserver:24.02-py3-min

ARG BUILD_TYPE=all
ENV DEBIAN_FRONTEND=noninteractive

RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt update -y \
    && apt install software-properties-common -y \
    && apt install python3 python3-pip -y \
    && apt install curl git sudo libibverbs-dev -y \
    && apt install -y rdma-core infiniband-diags ibverbs-utils openssh-server perftest ibverbs-providers libibumad3 libibverbs1 libnl-3-200 libnl-route-3-200 librdmacm1 \
    && python3 --version \
    && python3 -m pip --version \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

RUN pip3 install --upgrade pip
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# For openbmb/MiniCPM models
RUN pip3 install datamodel_code_generator --break-system-packages

WORKDIR /sgl-workspace

ARG CUDA_VERSION
COPY . /sgl-workspace/sglang
RUN python3 -m pip install --upgrade pip setuptools wheel html5lib six --break-system-packages --ignore-installed\
    # && git clone --depth=1 https://github.com/sgl-project/sglang.git \
    && if [ "$CUDA_VERSION" = "12.1.1" ]; then \
         export CUINDEX=121; \
       elif [ "$CUDA_VERSION" = "12.4.1" ]; then \
         export CUINDEX=124; \
       elif [ "$CUDA_VERSION" = "12.8.1" ]; then \
         export CUINDEX=124; \
       elif [ "$CUDA_VERSION" = "11.8.0" ]; then \
         export CUINDEX=118; \
         python3 -m pip install --no-cache-dir sgl-kernel -i https://docs.sglang.ai/whl/cu118 --break-system-packages; \
       else \
         echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1; \
       fi \
    && python3 -m pip install --no-cache-dir torch==2.7.1 --break-system-packages \
    && cd sglang \
    && python3 -m pip --no-cache-dir install -e "python[${BUILD_TYPE}]" --break-system-packages \
    && if [ "$CUDA_VERSION" = "12.8.1" ]; then \
         python3 -m pip install nvidia-nccl-cu12==2.26.2.post1 --force-reinstall --no-deps --break-system-packages; \
       fi

ENV DEBIAN_FRONTEND=interactive
