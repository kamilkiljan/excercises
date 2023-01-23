#!/bin/sh
cp $1 ./input_data.txt
IMAGE=$(docker build -q .)
docker run --name count_islands --rm $IMAGE
rm ./input_data.txt