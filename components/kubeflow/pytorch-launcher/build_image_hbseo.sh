#!/bin/bash

FULL_NAME=hbseo-regi.clova.ai/train/pytorch-launcher:0.3

mkdir -p ./build
rsync -arvp ./src/ ./build/
rsync -arvp ../common/ ./build/

echo "Building image $FULL_NAME"
docker build -t ${FULL_NAME} .

echo "Pushing image $FULL_NAME"
docker push ${FULL_NAME}

rm -rf ./build
