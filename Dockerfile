FROM pytorch/torchserve:latest

COPY models/exp1/config.json models/exp1/pytorch_model.bin src/models/model_handler.py /home/model-server/

USER root
RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
USER model-server

RUN torch-model-archiver --model-name deployed_model \
    --version 0.1 \
    --serialized-file /home/model-server/pytorch_model.bin \
    --export-path /home/model-server/model-store \
    --handler /home/model-server/model_handler.py \
    --extra-files /home/model-server/config.json

CMD ["torchserve", \
     "--start" \
     "--ts-config=/home/model-server/config.properties", \
     "--model", \
     "deployed_model=deployed_model.mar"]