#!/bin/bash
python3 main.py --eval \
                --data_path ../Data/CelebA \
                --image_channel 3 \
                --batch_size 10 \
                --output_path output/CelebA