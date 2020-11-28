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

CUDA = True     # Change to False for CPU training
DATA_PATH = './Data/mnist'
# DATA_PATH = '/media/john/FastData/CelebA'
# DATA_PATH = '/media/john/FastData/lsun'
OUT_PATH = 'output'
LOG_FILE = os.path.join(OUT_PATH, 'log.txt')
BATCH_SIZE = 128  # Adjust this value according to your GPU memory
IMAGE_CHANNEL = 1 # For gray scale
# IMAGE_CHANNEL = 3
Z_DIM = 100
G_HIDDEN = 64
X_DIM = 64
D_HIDDEN = 64
EPOCH_NUM = 25
REAL_LABEL = 1
FAKE_LABEL = 0
lr = 2e-4
seed = 1  # Change to None to get different results at each run


utils.clear_folder(OUT_PATH) # Empty the output folder for us and create one if it doesn't exist. 
print("Logging to {}\n".format(LOG_FILE))
sys.stdout = utils.StdOut(LOG_FILE) # Redirect all messages from print to the log file and show these messages in the console at the same time.
CUDA = CUDA and torch.cuda.is_available()
print("PyTorch version: {}".format(torch.__version__))
if CUDA:
    print("CUDA version: {}\n".format(torch.version.cuda))

if seed is None:
    seed = np.random.randint(1, 10000)
print("Random Seed: ", seed)
np.random.seed(seed)
torch.manual_seed(seed)
if CUDA:
    torch.cuda.manual_seed(seed)
cudnn.benchmark = True      # May train faster but cost more memory

dataset = datasets.MNIST(root=DATA_PATH, download=True,
                        transform=transforms.Compose([
                        transforms.Resize(X_DIM),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                     ]))
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=4)

device = torch.device("cuda:0" if CUDA else "cpu")

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
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 1st layer
            nn.ConvTranspose2d(in_channels = Z_DIM, out_channels = G_HIDDEN * 8, kernel_size = 4, stride = 1, padding = 0, bias=False),
            nn.BatchNorm2d(num_features = G_HIDDEN * 8),
            nn.ReLU(True),
            # 2nd layer
            nn.ConvTranspose2d(in_channels = G_HIDDEN * 8, out_channels = G_HIDDEN * 4, kernel_size = 4, stride = 1, padding = 0, bias=False),
            nn.BatchNorm2d(num_features = G_HIDDEN * 4),
            nn.ReLU(True),
            # 3rd layer
            nn.ConvTranspose2d(in_channels = G_HIDDEN * 4, out_channels = G_HIDDEN * 2, kernel_size = 4, stride = 1, padding = 0, bias=False),
            nn.BatchNorm2d(num_features = G_HIDDEN * 2),
            nn.ReLU(True),
            # 4th layer
            nn.ConvTranspose2d(in_channels = G_HIDDEN * 2, out_channels = G_HIDDEN, kernel_size = 4, stride = 1, padding = 0, bias=False),
            nn.BatchNorm2d(num_features = G_HIDDEN ),
            nn.ReLU(True),
            # output layer
            nn.ConvTranspose2d(G_HIDDEN, IMAGE_CHANNEL, kernel_size = 4,stride = 1, padding = 0, bias =False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 1st layer
            nn.Conv2d(IMAGE_CHANNEL, D_HIDDEN, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd layer
            nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd layer
            nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th layer
            nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # output layer
            nn.Conv2d(D_HIDDEN * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)