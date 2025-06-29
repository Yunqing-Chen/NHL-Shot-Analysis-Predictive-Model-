#!/bin/bash

# echo "TODO: fill in the docker run command"

docker run --name serving-client \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -p 5000:5000 \
    flask-serving-client

docker run --name streamlit-client \
    -p 8501:8501 \
    streamlit-client