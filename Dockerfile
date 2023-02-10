FROM tensorflow/tensorflow:latest-devel-gpu
WORKDIR /work
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
