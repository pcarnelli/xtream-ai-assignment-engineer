# How to run

## Build image

From this directory (`docker/challenge3`) run:
    
    $ docker build -t test/challenge3 -f ./Dockerfile ../../

## Run Flask REST API

    $ docker run --rm -it --name challenge3 -p 5000:5000 test/challenge3 python src/app.py

## Request from new TTY

Get the id of the container running the Flask REST API:

    $ docker ps

Run a `bash` shell in the same container:

    $ docker exec -it <container_id> bash

Run script `request.py` in the new shell from the same container:

    $ python src/request.py

Check output on screen and file `prediction.json` in the directory `res`.
