import os
from dcgan import *
from utils import *
from scipy.interpolate import interp1d
from datetime import datetime

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
        
        viz_tensor = torch.randn(params['batch_size'], params['z_dim'], 1, 1, device = params['device'])
        with torch.no_grad():
            viz_sample = netG(viz_tensor)
            cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            vutils.save_image(viz_sample, params['output_path']+'/img_{}.png'.format(cur_time), nrow = 10, normalize = True)