# FROM google/cloud-sdk:alpine as gcloud
FROM anibali/pytorch:1.8.1-cuda11.1-ubuntu20.04

# Copying files to files
COPY requirements.txt requirements.txt
COPY requirements_devel.txt requirements_devel.txt

COPY setup.py setup.py
COPY src/ src/

# Installing dependencies
RUN pip install -r requirements.txt
RUN pip install -r requirements_devel.txt

ARG WANDB_TOKEN=default
RUN echo "$(WANDB_TOKEN)"
ENV "WANDB_API_KEY"=WANDB_TOKEN

# Download data
RUN mkdir raw_data
RUN python src/data/download_data.py raw_data

RUN mkdir -p data/processed
RUN python src/data/make_dataset.py raw_data data/processed
RUN rm -rf raw_data

# Command run when doing docker run
ENTRYPOINT [ "python", "-u", "src/models/train_model.py" ]
