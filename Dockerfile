ARG BASE_IMAGE_NAME
FROM ${BASE_IMAGE_NAME}

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libsm6 \
        libxext6 \
        libxrender-dev

COPY requirements.txt /tmp/
COPY requirements_torch.txt /tmp/

RUN pip install -r /tmp/requirements.txt
RUN pip install -r /tmp/requirements_torch.txt

# Set working directory
WORKDIR /app

CMD ["/bin/bash"]

