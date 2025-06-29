#!/bin/bash

# echo "TODO: fill in the docker build command"
docker build -t flask-serving-client -f Dockerfile.serving .

docker build -t streamlit-client -f Dockerfile.streamlit .