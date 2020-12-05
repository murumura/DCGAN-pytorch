import os
from dcgan import *
from utils import *
from scipy.interpolate import interp1d
class Evaluator():
    def __init__(self, params):
        super().__init__()
        self.Generator_param_path = os.path.join(params['output_path'], 'netG_{}.pth'.format(params['nepochs']-1))
    def eval_algorithm(self, params):
        print(self.Generator_param_path)
        if os.path.isfile(self.Generator_param_path):
            netG = Generator(params).to(params['device'])
            netG.load_state_dict(torch.load(self.Generator_param_path))
        else:
            print ("Generator's weight does not exist")
        
        create_gif(params['nepochs'],params['output_path'])