from .blender import BlenderDataset
from .llff import LLFFDataset
from .llff_for_attack import LLFFDataset_attack
from .blender_for_attack import BlenderDataset_attack
from .opts import get_opts

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'llff_for_attack': LLFFDataset_attack,
                'blender_for_attack': BlenderDataset_attack
                }