import random
from pathlib import Path

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
    Path(name).parent.mkdir(exist_ok=True, parents=True)
    imageio.mimsave(name, frames, fps=fps)
    pygifsicle.optimize(name)
