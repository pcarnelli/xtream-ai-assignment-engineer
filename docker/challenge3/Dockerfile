# Credit: Ben Postance https://bpostance.github.io/posts/docker-fask-api/

# Pull the image from docker hub
FROM python:3.11.9

# Set WORKDIR - Working directory for any RUN, CMD, ENTRYPOINT, COPY and ADD instructions that follow it in the Dockerfile
WORKDIR /home/challenge3

# RUN - Create directories
RUN mkdir models
RUN mkdir src
RUN mkdir res

# COPY - Copy files or directories
COPY requirements.txt ./
COPY models/pipeline.joblib ./models/
COPY src/app.py ./src/
COPY src/request.py ./src/
COPY res/payload.json ./res/

# RUN - Install Python dependencies
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# EXPOSE - informs Docker that the container listens on the specified network ports at runtime
EXPOSE 5000
