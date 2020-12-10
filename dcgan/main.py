from dcgan import *
from train import Trainer
from eval import Evaluator
import argparse
def run(args):
    config_params(args)
    init()
    if(args.train):
        gan_trainer = Trainer(params)
        gan_trainer.train(params)
    elif(args.eval):
        gan_eval = Evaluator(params)
        gan_eval.eval_algorithm(params)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--train', action='store_true')
    p.add_argument('--eval', action='store_true')
    p.add_argument("--image_channel", type = int, default = 1)
    p.add_argument("--batch_size", type = int, default = 128)
    p.add_argument("--data_path", type=str, default="../Data/mnist")
    p.add_argument("--dataset", type=str, default="mnist")
    p.add_argument("--output_path", type=str, default = "output/mnist")
    p.add_argument("--output_log", type = str, default = "output/mnist/log.txt")
    args = p.parse_args()
    run(args)