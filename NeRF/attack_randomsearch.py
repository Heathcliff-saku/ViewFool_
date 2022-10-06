"""
Random view sampling and rendering (Make sure the '--optim_method' in the command is 'random')

"""

from rendering_image import render_image
import numpy as np
from PIL import Image
import joblib
import torch
from datasets.opts import get_opts
import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(current_directory) + os.path.sep + ".")
sys.path.append(root_path)
import classifier.predict 
from classifier.predict import test_baseline 

import xlrd
import numpy as np


# random search 
x = render_image([0, 0, -30, 4.0, 0, 0])

# test_baseline(path="C:/Users/Silvester/PycharmProjects/NeRFAttack/NeRF/results/blender_for_attack/'hotdog'/", label='hotdog, hot dog, red hot')
