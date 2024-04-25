# How to run

## Build image

From this directory (`docker/challenge2`) run:

    $ docker build -t test/challenge2 -f ./Dockerfile ../../

## Interactive shell

Start container and run `bash` shell:

    $ docker run --rm -it --name challenge2 test/challenge2 bash

Run script `train.py` in the container:

    $ python src/steps/train.py

Check outputs on screen and in the directory `models`.
