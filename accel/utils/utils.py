import random

import imageio
import numpy as np
import pygifsicle
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_as_video(name, frames, fps=60):
    imageio.mimsave(name, frames, fps=fps)
    pygifsicle.optimize(name)
