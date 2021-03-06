# Common imports
import numpy as np
from pathlib import Path
from PIL import Image
import imageio
#To plot pretty picture
import matplotlib as mpl
import matplotlib.pyplot as plt
import errno
import shutil
import sys
import os
import torch
import torch.utils.data as data
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
n_steps = 50
# Where to save the figures
PROJECT_ROOT_DIR = ".."
CHAPTER_ID = "DCGAN"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout = True, fig_extension = "png", resolution = 300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format = fig_extension, dpi = resolution)
'''
create gif of plotted loss, synsethesis images
'''
def create_gif(epoches, save_dir):
    images = []
    for i in range(1, epoches + 1):
        images.append(imageio.imread(os.path.join(save_dir, 'fake_samples_{}.png'.format(i))))
    imageio.mimsave(os.path.join(save_dir, 'result.gif'), images, fps=5)
    images = []
    for i in range(1, epoches + 1):
        images.append(imageio.imread(os.path.join(save_dir, 'DCGAN_loss_epoch_{}.png'.format(i))))
    imageio.mimsave(os.path.join(save_dir, 'result_loss.gif'), images, fps=5)


"""
Exports torch.Tensor to Numpy array.
"""
def to_np(var):
    return var.detach().cpu().numpy()

"""
    Create a folder if it does not exist.
"""
def create_folder(folder_path):
    try:
        os.makedirs(folder_path)
    except OSError as _e:
        if _e.errno != errno.EEXIST:
            raise

"""
    Clear all contents recursively if the folder exists.
    Create the folder if it has been accidently deleted.
"""
def clear_folder(folder_path):
    
    create_folder(folder_path)
    for the_file in os.listdir(folder_path):
        _file_path = os.path.join(folder_path, the_file)
        try:
            if os.path.isfile(_file_path):
                os.unlink(_file_path)
            elif os.path.isdir(_file_path):
                shutil.rmtree(_file_path)
        except OSError as _e:
            print(_e)

"""
Redirect stdout to file, and print to console as well.
"""
class StdOut(object):    
    def __init__(self, output_file):
        self.terminal = sys.stdout
        self.log = open(output_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def plot_loss(d_loss, g_loss, num_epoch, epoches, save_dir):
    
    fig, ax = plt.subplots()
    ax.set_xlim(0,epoches + 1)
    ax.set_ylim(0, max(np.max(g_loss), np.max(d_loss)) * 1.1)
    plt.xlabel('Epoch {}'.format(num_epoch))
    plt.ylabel('Loss')
    
    plt.plot([i for i in range(1, num_epoch + 1)], d_loss, label='Discriminator', color='red', linewidth=3)
    plt.plot([i for i in range(1, num_epoch + 1)], g_loss, label='Generator', color='mediumblue', linewidth=3)
    
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'DCGAN_loss_epoch_{}.png'.format(num_epoch)))
    plt.close()

def read_data_file(filename, root=None):
    
    lists = []
    with open(filename, 'r') as fp:
        line = fp.readline()
        while line:
            info = line.strip().split(' ')
            if root is not None:
                info[0] = os.path.join(root, info[0])
            if len(info) == 1:
                item = (info[0], 1)
            else:
                item = (info[0], int(info[1]))
            lists.append(item)
            line = fp.readline()
    
    return lists

"""
 A data loader where the list is arranged in this way:
        
        dog/1.jpg 1
        dog/2.jpg 1
        dog/3.jpg 1
            .
            .
            .
        cat/1.jpg 2
        cat/2.jpg 2
            .
            .
            .
        path      label
    
    Args:
      
        root (string): Root directory path.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version
"""
class MyDataFolder(data.Dataset): 
    def __init__(self, filename, root = None, transform = None):

        lists = read_data_file(filename)
        if len(lists) == 0:
            raise(RuntimeError('Found 0 images in subfolders\n'))

        self.root = root
        self.transform = transform
        self.lists = lists
        self.load = lambda path : Image.load(path).convert('RGB')

    def __getitem__(self, index):
        """
        Args:
            index (int): index
    
        Returns:
            tuple: (image, label) where label is the clas of the image
        """
        path, label = self.lists[index]
        img = self.load(path)
    
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label
    
    def __len__(self):
        
        return len(self.lists)