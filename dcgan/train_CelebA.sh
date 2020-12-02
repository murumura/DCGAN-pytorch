#!/bin/bash
python3 main.py --train \
                --data_path ../Data/CelebA \
                --image_channel 3 \
                --dataset CelebA \
                --output_path output/CelebA \
                --output_log output/CelebA/log.txt