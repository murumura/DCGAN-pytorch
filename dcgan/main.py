from dcgan import *
from train import Trainer
import argparse
def run(args):
    config_params(args)
    init()
    gan_trainer = Trainer(params)
    gan_trainer.train(params)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--train', action='store_true')
    p.add_argument('--eval', action='store_true')
    args = p.parse_args()
    run(args)