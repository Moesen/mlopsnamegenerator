# FROM google/cloud-sdk:alpine as gcloud
FROM anibali/pytorch:1.8.1-cuda11.1-ubuntu20.04

ARG WANDB_TOKEN
ENV "WANDB_API_KEY" "$WANDB_TOKEN"

