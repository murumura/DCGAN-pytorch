#!/bin/bash
docker run --gpus all  -it -v $(pwd):/torch -p 1234:8888 -p 6006:6006 --name torch_container torch-dev:latest
