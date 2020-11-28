import os
from dcgan import *
class Trainer():
    def __init__(self, params):
        super().__init__()
        self.save_epoch = params['save_epoch']
        self.num_epochs = params['nepochs'] 
        self.output_path = params['output_path']
    def train(self, params):
        netG = Generator(params).to(params['device'])
        netG.apply(weights_init)
        print(netG)
        netD = Discriminator(params).to(params['device'])
        netD.apply(weights_init)
        print(netD)
        criterion = nn.BCELoss() 
         