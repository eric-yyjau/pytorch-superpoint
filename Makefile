RELEASE?=0
TAG?=1
IMAGE_NAME?=visual_navigation_superpoint
BASE_IMAGE_NAME?=python:3.6

RELEASE_TAG=$(RELEASE).$(TAG)
IMAGE_NAME_LOCAL=$(IMAGE_NAME):$(RELEASE_TAG)

WORKDIR := $(shell pwd)


build: ## build image for dev, which is used for development
	echo $(IMAGE_NAME_LOCAL)
	echo $(BASE_IMAGE_NAME)
	docker build --build-arg BASE_IMAGE_NAME=$(BASE_IMAGE_NAME) . -t $(IMAGE_NAME_LOCAL) -f Dockerfile

run_container:
	echo $(IMAGE_NAME_LOCAL)
	docker run -it -d --gpus all --shm-size=4gb \
	-v "$(WORKDIR)":/app \
	--name superpoint_train \
	$(IMAGE_NAME_LOCAL)