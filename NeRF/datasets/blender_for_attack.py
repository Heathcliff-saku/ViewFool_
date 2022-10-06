import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *
from .opts import get_opts

def pose_spherical(all_args):
    args = get_opts()
    search_num = args.search_num

    if search_num == 3:
        theta, phi, radius = all_args
    if search_num == 6 or search_num == 123 or search_num == 456:
        gamma, theta, phi, radius, x, y = all_args

    trans_t = lambda t, x, y: torch.Tensor([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, t],
        [0, 0, 0, 1]]).float()

    rot_phi = lambda phi: torch.Tensor([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]]).float()

    rot_gamma = lambda gamma: torch.Tensor([
        [np.cos(gamma), 0, -np.sin(gamma), 0],
        [0, 1, 0, 0],
        [np.sin(gamma), 0, np.cos(gamma), 0],
        [0, 0, 0, 1]]).float()

    rot_theta = lambda th: torch.Tensor([
        [np.cos(th), -np.sin(th), 0, 0],
        [np.sin(th), np.cos(th), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]).float()

    if search_num == 3:
        c2w = trans_t(radius)
        c2w = rot_phi(phi / 180. * np.pi) @ c2w
        c2w = rot_theta(theta / 180. * np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w

    if search_num == 6 or search_num == 123 or search_num == 456:
        c2w = trans_t(radius, x, y)
        c2w = rot_phi(phi / 180. * np.pi) @ c2w
        c2w = rot_gamma(gamma / 180. * np.pi) @ c2w
        c2w = rot_theta(theta / 180. * np.pi) @ c2w

    return c2w


class BlenderDataset_attack(Dataset):
    def __init__(self, root_dir, all_args, is_over, split='train', spheric_poses=False, img_wh=(800, 800)):
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()

        self.read_meta()
        self.white_back = True
        self.spheric_poses = spheric_poses

        self.all_args = all_args
        self.is_over = is_over


    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
                                                                     # when W=800

        self.focal *= self.img_wh[0]/800 # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal) # (h, w, 3)


    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        args = get_opts()
        if args.optim_method == 'random':
            return args.num_sample
        else:
            if self.is_over:
                a = args.num_sample+1
            else:
                a = 1
            return a

    def __getitem__(self, idx):
        #frame = self.meta['frames'][idx]
        #c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]
        # c2w = torch.FloatTensor(self.poses_test[idx])

        # c2w = [[-1.0, 0.0, 0.0, 0.0],
        #       [0.0, -0.7341099977493286, 0.6790306568145752, 2.737260103225708],
        #       [0.0, 0.6790306568145752, 0.7341099381446838, 2.959291696548462]]
        #
        # poses = []
        # # a_ = np.pi/10 * np.random.random(self.num_sample)
        #
        # for a in np.linspace(0, 2*np.pi, self.num_sample):
        #     # c2w = trans_martix(0, np.pi, phi) @ c2w
        #     # phi = 2 * np.pi * np.random.random(1)
        #     # phi = phi[0]
        #     pose = trans_martix(a, np.pi, 0, t=6) @ c2w
        #     poses += [pose]
        # c2w = torch.FloatTensor(poses[idx])

        args = get_opts()
        mode = args.search_index
        method = args.optim_method

        if method == 'center_random':
            render_poses_ = torch.tensor([])
            gamma_c, th_c, phi_c, r_c, a_c, b_c = self.all_args
            random = np.zeros([args.num_sample, 6])

            gamma = 0.5*epsilon*60*(2*np.random.random(args.num_sample)-1) + gamma_c
            th = 0.5*epsilon*360*(2*np.random.random(args.num_sample)-1) + th_c
            phi = 0.5*epsilon*140*(2*np.random.random(args.num_sample)-1) + phi_c
            r = 0.5*epsilon*2*(2*np.random.random(args.num_sample)-1) + r_c
            a = 0.5*epsilon*1*(2*np.random.random(args.num_sample)-1) + a_c
            b = 0.5*epsilon*1*(2*np.random.random(args.num_sample)-1) + b_c

            random[:, 0] = gamma
            random[:, 1] = th
            random[:, 2] = phi
            random[:, 3] = r
            random[:, 4] = a
            random[:, 5] = b

            render_poses_ = torch.stack([pose_spherical([gamma_, th_, phi_, r_, a_, b_]) for gamma_, th_, phi_, r_, a_, b_ in random], 0)



        if method == 'random':
            render_poses_ = torch.tensor([])
            th_low, th_up = args.th_range
            phi_low, phi_up = args.phi_range
            r_low, r_up = args.r_range

            if args.search_num == 3:
                random = np.zeros([args.num_sample, 2])
                random_th = (th_up-th_low) * np.random.random(args.num_sample) + th_low
                random_phi = (phi_up-phi_low) * np.random.random(args.num_sample) + phi_low
                random[:, 0] = random_th
                random[:, 1] = random_phi
                render_poses_ = torch.stack([pose_spherical([a, b, 4.0]) for a, b in random], 0)

            ''' gamma, theta, phi, radius, x, y '''
            if args.search_num == 6:
                random = np.zeros([args.num_sample, 6])

           
                gamma = 60 * np.random.random(args.num_sample) - 30
                th = 360 * np.random.random(args.num_sample) - 180
                phi = 140 * np.random.random(args.num_sample) - 70
                r = 2 * np.random.random(args.num_sample) + 3
                a = np.random.random(args.num_sample) - 0.5
                b = np.random.random(args.num_sample) - 0.5

                random[:, 0] = gamma
                random[:, 1] = th
                random[:, 2] = phi
                random[:, 3] = r
                random[:, 4] = a
                random[:, 5] = b

                render_poses_ = torch.stack([pose_spherical([gamma_, th_, phi_, r_, a_, b_]) for gamma_, th_, phi_, r_, a_, b_ in random], 0)


        else: #NES
            if self.is_over:
                random = self.all_args  # If the search is complete, 100 images of the random are generated
                render_poses_ = torch.stack([pose_spherical([gamma_, th_, phi_, r_, a_, b_]) for gamma_, th_, phi_, r_, a_, b_ in random], 0)

            else:
                render_poses_ = torch.stack([pose_spherical(self.all_args)], 0) # In the normal iteration process, only one image of the corresponding angle is generated




        # render_poses_ = torch.stack([pose_spherical(angle, -30, 4.0) for angle in np.linspace(0, -180, self.num_sample+1)[:-1]], 0)
        # render_poses_ = torch.stack([pose_spherical(-90, -30, z) for z in np.linspace(1.0, 8.0, self.num_sample + 1)[:-1]], 0)

        c2w = render_poses_[idx, :3, :]

        # img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
        # img = img.resize(self.img_wh, Image.LANCZOS)
        # img = self.transform(img) # (4, H, W)
        # valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
        # img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
        # img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

        rays_o, rays_d = get_rays(self.directions, c2w)

        rays = torch.cat([rays_o, rays_d,
                            self.near*torch.ones_like(rays_o[:, :1]),
                            self.far*torch.ones_like(rays_o[:, :1])],
                            1) # (H*W, 8)

        sample = {'rays': rays,
                    'c2w': c2w}

        return sample