import os
import cv2

from collections import defaultdict

import torch
from tqdm import tqdm
import imageio


from models.rendering import render_rays
from models.nerf import *

from utils import load_ckpt
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *

torch.backends.cudnn.benchmark = True
from datasets.opts import get_opts


def render_image(all_args, is_over=False):

    @torch.no_grad() 
    def batched_inference(models, embeddings,
                        rays, N_samples, N_importance, use_disp,
                        chunk):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, chunk):
            rendered_ray_chunks = \
                render_rays(models,
                            embeddings,
                            rays[i:i+chunk],
                            N_samples,
                            use_disp,
                            0,
                            0,
                            N_importance,
                            chunk,
                            dataset.white_back,
                            test_time=True)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v.cpu()]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results


    args = get_opts()
    w, h = args.img_wh

    kwargs = {'root_dir': args.root_dir,
              'split': args.split,
              'img_wh': tuple(args.img_wh),
              'all_args': all_args,
              'is_over': is_over
              }

    kwargs['spheric_poses'] = args.spheric_poses
    dataset = dataset_dict[args.dataset_name](**kwargs)


    embedding_xyz = Embedding(args.N_emb_xyz)
    embedding_dir = Embedding(args.N_emb_dir)


    nerf_coarse = NeRF(in_channels_xyz=6*args.N_emb_xyz+3,
                       in_channels_dir=6*args.N_emb_dir+3)
    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    nerf_coarse.cuda().eval()

    models = {'coarse': nerf_coarse}
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}


    if args.N_importance > 0:
        nerf_fine = NeRF(in_channels_xyz=6*args.N_emb_xyz+3,
                         in_channels_dir=6*args.N_emb_dir+3)
        load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
        nerf_fine.cuda().eval()
        models['fine'] = nerf_fine       


    imgs, depth_maps, psnrs = [], [], []
    dir_name = f'results/{args.dataset_name}/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)

    
    for i in range(len(dataset)):
        sample = dataset[i]
        rays = sample['rays'].cuda()
        results = batched_inference(models, embeddings, rays,
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk)


        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        img_pred = np.clip(results[f'rgb_{typ}'].view(h, w, 3).cpu().numpy(), 0, 1)

        if args.save_depth:
            depth_pred = results[f'depth_{typ}'].view(h, w).cpu().numpy()
            depth_maps += [depth_pred]
            if args.depth_format == 'pfm':
                save_pfm(os.path.join(dir_name, f'depth_{i:03d}.pfm'), depth_pred)
            else:
                with open(os.path.join(dir_name, f'depth_{i:03d}'), 'wb') as f:
                    f.write(depth_pred.tobytes())

        img_pred_ = (img_pred * 255).astype(np.uint8)
        imgs += [img_pred_]
        imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)
    # imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}.gif'), imgs, fps=10)
    return img_pred_




    


