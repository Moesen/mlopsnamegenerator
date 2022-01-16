# cuda enabled python image
FROM anibali/pytorch:1.8.1-cuda11.1-ubuntu20.04

# Copying files to files
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

# Changing workdir to /app as that is expected in the anibali image
WORKDIR /app

# Installing dependencies
RUN pip install -r requirements.txt

# Environment variable for wandb
ENV "WANDB_API_KEY" "3e2d878ba2fb140799e656b41f70fd79182cf284"

# Command run when doing docker run
ENTRYPOINT [ "python", "-u", "src/models/predict_models.py"]