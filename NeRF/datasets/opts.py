from argparse import ArgumentParser

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff', 'llff_for_attack', 'blender_for_attack'],
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output folder name')
    parser.add_argument('--split', type=str, default='test',
                        help='test or test_train')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--N_emb_xyz', type=int, default=10,
                        help='number of frequencies in xyz positional encoding')
    parser.add_argument('--N_emb_dir', type=int, default=4,
                        help='number of frequencies in dir positional encoding')
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--chunk', type=int, default=32*1024*4,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')

    parser.add_argument('--save_depth', default=False, action="store_true",
                        help='whether to save depth prediction')
    parser.add_argument('--depth_format', type=str, default='pfm',
                        choices=['pfm', 'bytes'],
                        help='which format to save')

    parser.add_argument('--search_index', type=str,
                        default='th_phi',
                        choices=['th_phi', 'th', 'phi', 'r'],
                        help='search_index')
    parser.add_argument('--th_range', nargs="+", type=int, default=[-180, 180],
                        help='th_range')
    parser.add_argument('--phi_range', nargs="+", type=int, default=[-180, 0],
                        help='phi_range')
    parser.add_argument('--r_range', nargs="+", type=int, default=[4, 6],
                        help='r_range')
    parser.add_argument('--num_sample', type=int, default=100,
                        help='num_sample')

    parser.add_argument('--optim_method', type=str, default='random',
                        choices=['random', 'NES', 'xNES'],
                        help='num_sample')

    parser.add_argument('--search_num', type=int, default=3,
                        help='search_num')

    parser.add_argument('--target_flag', type=bool, default=False,
                        help='target_flag')

    parser.add_argument('--target_label', type=int, default=584,
                        help='target_label')

    parser.add_argument('--popsize', type=int, default=21,
                        help='popsize')

    parser.add_argument('--iteration', type=int, default=20,
                        help='iteration')

    parser.add_argument('--mu_lamba', type=float, default=0.0001,
                        help='iteration')
    parser.add_argument('--sigma_lamba', type=float, default=0.0001,
                        help='iteration')

    parser.add_argument('--random_eplison', type=float, default=0.01,
                        help='iteration')

    parser.add_argument('--index', type=float, default=0,
                        help='iteration')


    parser.add_argument('--label_name', type=str, default='hotdog, hot dog, red hot',help='The correct label for the current attack, or the target label if it is targeted')

    parser.add_argument('--label', type=int, default=934,help='The correct label for the current attack, or the target label if it is targeted')



    return parser.parse_args()

