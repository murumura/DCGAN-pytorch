import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

import utils


# Parameters to define the model.
params = {
    'gpu' : True,           # Using GPU to train the model
    "batch_size" : 128,     # Batch size during training.
    'imsize' : 64,          # Spatial size of training images. All images will be resized to this size during preprocessing.
    'image_channel' : 1,    # Number of channles in the training images. For coloured images this is 3.
    'z_dim' : 100,          # Size of the Z latent vector (the input to the generator).
    'ngf' : 64,             # Size of feature maps in the generator. The depth will be multiples of this.
    'ndf' : 64,             # Size of features maps in the discriminator. The depth will be multiples of this.
    'nepochs' : 25,         # Number of training epochs.
    'lr' : 0.0002,          # Learning rate for optimizers
    'beta1' : 0.5,          # Beta1 hyperparam for Adam optimizer
    'save_epoch' : 2,       # Save step.
    'cuda_dnn_benchmark' : True,
    'seed' : 1,
    'data_path' : '../Data/mnist',
    'output_path' : 'output',
    'output_log' : os.path.join('output', 'log.txt')
}
# Establish convention for real and fake labels during training
REAL_LABEL = 1
FAKE_LABEL = 0

def config_params(args):
    for arg in vars(args):
        print(arg, getattr(args, arg))
def init():
    # Empty the output folder for us and create one if it doesn't exist.
    utils.clear_folder(params['output_path'])  
    print("Logging to {}\n".format(params['output_log']))
    # Redirect all messages from print to the log file and show these messages in the console at the same time.
    sys.stdout = utils.StdOut(params['output_log']) 
    CUDA = params['gpu']
    CUDA = CUDA and torch.cuda.is_available()
    print("PyTorch version: {}".format(torch.__version__))
    if CUDA:
        print("CUDA version: {}\n".format(torch.version.cuda))
    seed = params['seed']
    if params['seed'] is None:
        seed = np.random.randint(1, 10000)

    print("Random Seed: ", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if CUDA:
        torch.cuda.manual_seed(seed)
    params.update({'device': torch.device("cuda:0" if CUDA else "cpu")})
    cudnn.benchmark = params['cuda_dnn_benchmark']      # May train faster but cost more memory

"""
 custom weights initialization:
 we initialize the convolution kernels based on the Gaussian distribution (normal distribution)
 with a mean of 0 and a standard deviation of 0.02. 
 We also need to initialize the affine parameters (scaling factors) in batch normalization
"""
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 1st layer
            nn.ConvTranspose2d(in_channels = params['z_dim'], out_channels = params['ngf'] * 8, kernel_size = 4, stride = 1, padding = 0, bias=False),
            nn.BatchNorm2d(num_features = params['ngf'] * 8),
            nn.ReLU(True),
            # 2nd layer
            nn.ConvTranspose2d(in_channels = params['ngf'] * 8, out_channels = params['ngf'] * 4, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(num_features = params['ngf'] * 4),
            nn.ReLU(True),
            # 3rd layer
            nn.ConvTranspose2d(in_channels = params['ngf'] * 4, out_channels = params['ngf'] * 2, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(num_features = params['ngf'] * 2),
            nn.ReLU(True),
            # 4th layer
            nn.ConvTranspose2d(in_channels = params['ngf'] * 2, out_channels = params['ngf'], kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(num_features = params['ngf'] ),
            nn.ReLU(True),
            # output layer
            nn.ConvTranspose2d(params['ngf'], params['image_channel'], kernel_size = 4,stride = 2, padding = 1, bias = False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 1st layer
            nn.Conv2d(params['image_channel'], params['ndf'], 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd layer
            nn.Conv2d(params['ndf'], params['ndf'] * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(params['ndf'] * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd layer
            nn.Conv2d(params['ndf'] * 2, params['ndf'] * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(params['ndf'] * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th layer
            nn.Conv2d(params['ndf'] * 4, params['ndf'] * 8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(params['ndf'] * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # output layer
            nn.Conv2d(params['ndf'] * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

"""
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.
"""
def get_mnist(params):
    dataset = datasets.MNIST(root= params['data_path'], download=True,
                     transform = transforms.Compose([
                     transforms.Resize(params['imsize']),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5,), (0.5,))
                     ]))
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size = params['batch_size'],
                                            shuffle = True,
                                            num_workers = 4)
    return dataloader