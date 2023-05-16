#!/bin/bash
docker run \
    --interactive --tty \
    --rm \
    --user $(id -u):$(id -g) \
    --cpus 32 \
    --gpus '"device=0,1"' \
    --volume $PWD:$PWD \
    --workdir $PWD \
    alessandroquerci/tirocinio:latest \
    python code_py/training.py --epochs=20 --decoder=fpn --backbone=resnet34 --train

