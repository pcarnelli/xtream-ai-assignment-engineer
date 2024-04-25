# Credit: Ben Postance https://bpostance.github.io/posts/docker-fask-api/

# Pull the image from docker hub
FROM python:3.11.9

# Set WORKDIR - Working directory for any RUN, CMD, ENTRYPOINT, COPY and ADD instructions that follow it in the Dockerfile
WORKDIR /home/challenge2

# RUN - Create directories
RUN mkdir models
RUN mkdir -p src/steps
RUN mkdir -p datasets/diamonds

# COPY - Copy files or directories
COPY requirements.txt ./
COPY src/steps/train.py ./src/steps/
COPY datasets/diamonds/diamonds.csv ./datasets/diamonds

# RUN - Install Python dependencies
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

