""" Fuctions/classes to be used during training process """
import os
import shutil
import torch
import collections
import threading
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

def save_checkpoint(state:dict, is_best:bool, save_dir:str, name:str='last.pth'):
    """
    Saves training checkpoint to disk

    Args:
        state: Model's state in dictionary
        is_best: Whether the model is the best (will save a best.pth file)
        save_dir: directory to save the checkpoint to
        name: Name of the checkpoint to be saved
    """
    # save the checkpoint to the file
    print('=> saving checkpoint', flush=True)
    file_name = os.path.join(save_dir, name)
    torch.save(state, file_name)

    # make a copy as `best.pth` if is the best checkpoint
    if is_best:
        shutil.copyfile(file_name, os.path.join(
            save_dir, 'best.pth'))

class AverageMeter(object):

    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.reset()
        self.lock = threading.Lock()

    def reset(self):
        self.sum = [0] * self.num_classes
        self.count = [0] * self.num_classes
        self.avg_per_class = [0] * self.num_classes
        self.avg = 0

    def update(self, val, cl=0):
        with self.lock:
            self.sum[cl] += val
            self.count[cl] += 1
            self.avg_per_class = [
                x/y if x > 0 else 0 for x, y in zip(self.sum, self.count)]
            self.avg = sum(self.avg_per_class)/len(self.avg_per_class)

class Logger:

    def __init__(self, keys, title=""):

        self.data = {k: [] for k in keys}
        self.title = title
        self.win = None

        print('Created logger with keys:  {}'.format(keys), flush=True)

    def plot(self, save=False, save_dir=""):

        if self.win is None:
            self.win = plt.subplots()
        fig, ax = self.win
        ax.cla()

        keys = []
        for key in self.data:#TODO earlier self.data
            keys.append(key)
            data = self.data[key]
            ax.plot(range(len(data)), data, marker='.')

        ax.legend(keys, loc='upper right')
        ax.set_title(self.title)
        plt.draw()
        Visualizer.mypause(0.001)

        if save:
            # save figure
            fig.savefig(os.path.join(save_dir, self.title + '.png'))

            # save data as csv
            df = pd.DataFrame.from_dict(self.data)
            df.to_csv(os.path.join(save_dir, self.title + '.csv'))

    def add(self, key, value):
        assert key in self.data, "Key not in data"
        self.data[key].append(value)


class Visualizer:

    def __init__(self, keys):
        self.wins = {k: None for k in keys}

    def display(self, image, key, title, title2=None):

        n_images = len(image) if isinstance(image, (list, tuple)) else 1

        if self.wins[key] is None:
            self.wins[key] = plt.subplots(ncols=n_images)

        fig, ax = self.wins[key]
        n_axes = len(ax) if isinstance(ax, collections.Iterable) else 1

        assert n_images == n_axes

        if n_images == 1:
            ax.cla()
            ax.set_axis_off()
            ax.imshow(self.prepare_img(image), cmap='magma')
            ax.set_title(title)
        else:
            for i in range(n_images):
                ax[i].cla()
                ax[i].set_axis_off()
                ax[i].imshow(self.prepare_img(image[i]), cmap='magma')
                if i == 0:
                    ax[i].set_title(title)
                elif i == 1:
                    ax[i].set_title(title2)
        plt.draw()
        self.mypause(0.001)

    @staticmethod
    def prepare_img(image):
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, torch.Tensor):
            image.squeeze_()
            image = image.numpy()

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] in {1, 3}:
                image = image.transpose(1, 2, 0)
            return image

    @staticmethod
    def mypause(interval):
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return