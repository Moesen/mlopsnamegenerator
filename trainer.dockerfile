# cuda enabled python image
FROM anibali/pytorch:1.8.1-cuda11.1-ubuntu20.04

# Copying files to files
COPY requirements.txt requirements.txt
COPY requirements_devel.txt requirements_devel.txt
COPY .dvc/ .dvc/
COPY data.dvc data.dvc

COPY setup.py setup.py
COPY src/ src/

# Changing workdir to /app as that is expected in the anibali image
WORKDIR /app

# Installing dependencies
RUN pip install -r requirements.txt
RUN pip install -r requirements_devel.txt

# Downloading data and 
RUN dvc pull

# Command run when doing docker run
ENTRYPOINT [ "python", "-u", "src/models/train_model.py" ]
