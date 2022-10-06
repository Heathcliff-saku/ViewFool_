from rendering_image import render_image
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt

import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(current_directory) + os.path.sep + ".")
sys.path.append(root_path)
import classifier.predict_3 
from classifier.predict_3 import test_baseline 

# from classifier import test_baseline

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import argparse


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='initial model')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

opt = parse_opt()

print(opt.model)

aa = ['hotdog', 'bread', 'burger', 'lemon', 'squash', 'mushroom', 'apple_2', 'simit', 'hotdog_2', 'burger_2', 'car_2', 'tank', 'wheel', 'wheel_2', 'scooter', 'forklift', 'bike', 'traffic_light', 'warplane', 'ship', 'sofa_2', 'crate', 'toaster', 'chair', 'bathing', 'sofa_3', 'piano', 'chair_2', 'washer', 'vase', 'mic', 'computer', 'mouse', 'camera', 'control', 'lamp', 'guitar', 'iron', 'keyboard', 'type', 'trash_bin', 'bottle', 'shoe', 'basketball', 'shoe_2', 'pool_table', 'sign', 'sign_2', 'sign_3', 'manhole', 'clock', 'clock_2', 'warplane_2', 'chair_3', 'car_3', 'sofa_4', 'glass', 'phone', 'guitar_2', 'volleyball', 'barrel', 'mic_2', 'scooter_2', 'tool', 'shoe_3', 'candle', 'coffeepot', 'vase_2', 'teapot', 'plane', 'car_4', 'soap', 'ipod', 'cassette', 'sofa_5', 'lip', 'sax', 'cart', 'cart_2', 'barrel_2', 'light', 'desk', 'coffee', 'cup', 'goblet', 'umbrella', 'helmet', 'syringe', 'swing', 'teddy', 'screwdriver', 'bank', 'lock', 'abacus', 'knife', 'hourglass', 'hammer', 'telephone', 'binoculars', 'barbell']


# aa = ['1.1hotdog', '1.2bread', '1.3burger', '1.4lemon', '1.5 squash', '1.6mushroom', '1.7apple_2', '1.8simit', '1.9hotdog_2', '1.1burger_2', '2.1car_2', '2.2tank', '2.3wheel', '2.4wheel_2', '2.5scooter', '2.6forklift', '2.7bike', '2.8traffic_light', '2.9warplane', '2.1ship', '3.1sofa_2', '3.2crate', '3.3toaster', '3.4chair', '3.5bathing', '3.6sofa_3', '3.7piano', '3.8chair_2', '3.9washer', '3.1vase', '4.1mic', '4.2computer', '4.3mouse', '4.4camera', '4.5control', '4.6lamp', '4.7guitar', '4.8iron', '4.9keyboard', '4.1type', '5.1trash_bin ', '5.2bottle', '5.3shoe', '5.4basketball', '5.5shoe_2', '5.6pool_table', '5.7sign', '5.8sign_2', '5.9sign_3', '5.1manhole', '6.1clock', '6.2clock_2', '6.3warplane_2', '6.4chair_3', '6.5car_3', '6.6sofa_4', '6.7glass', '6.8phone', '6.9guitar_2', '6.1volleyball', '6.11barrel', '6.12mic_2', '6.13scooter_2', '6.14tool', '6.15shoe_3', '6.16candle', '6.17coffeepot', '6.18vase_2', '6.19teapot', '6.2plane', '6.21car_4', '6.22soap', '6.23ipod', '6.24cassette', '6.25sofa_5', '6.26lip', '6.27sax', '6.28cart', '6.29cart_2', '6.3barrel_2', '6.31light', '6.32desk', '6.33coffee', '6.34cup', '6.35goblet', '6.36umbrella', '6.37helmet', '6.38syringe', '6.39swing', '6.4teddy', '6.41screwdriver', '6.42bank', '6.43lock', '6.44abacus', '6.45knife', '6.46hourglass', '6.47hammer', '6.48telephone', '6.49binoculars', '6.5barbell']

# aa = ['1.1hotdog', '1.2bread', '1.3burger', '1.4lemon', '1.5squash', '1.6mushroom', '1.7apple_2', '1.8simit', '1.9hotdog_2', '1.1burger_2', '2.1car_2', '2.2tank', '2.3wheel', '2.4wheel_2', '2.5scooter', '2.6forklift', '2.7bike', '2.8traffic_light', '2.9warplane', '2.1ship', '3.1sofa_2', '3.2crate', '3.3toaster', '3.4chair', '3.5bathing', '3.6sofa_3', '3.7piano', '3.8chair_2', '3.9washer', '3.1vase', '4.1mic', '4.2computer', '4.3mouse', '4.4camera', '4.5control', '4.6lamp', '4.7guitar', '4.8iron', '4.9keyboard', '4.1type', '5.1trash_bin', '5.2bottle', '5.3shoe', '5.4basketball', '5.5shoe_2', '5.6pool_table', '5.7sign', '5.8sign_2', '5.9sign_3', '5.1manhole', '6.1clock', '6.2clock_2', '6.3warplane_2', '6.4chair_3', '6.5car_3', '6.6sofa_4', '6.7glass', '6.8phone', '6.9guitar_2', '6.1volleyball', '6.11barrel', '6.12mic_2', '6.13scooter_2', '6.14tool', '6.15shoe_3', '6.16candle', '6.17coffeepot', '6.18vase_2', '6.19teapot', '6.2plane', '6.21car_4', '6.22soap', '6.23ipod', '6.24cassette', '6.25sofa_5', '6.26lip', '6.27sax', '6.28cart', '6.29cart_2', '6.3barrel_2', '6.31light', '6.32desk', '6.33coffee', '6.34cup', '6.35goblet', '6.36umbrella', '6.37helmet', '6.38syringe', '6.39swing', '6.4teddy', '6.41screwdriver', '6.42bank', '6.43lock', '6.44abacus', '6.45knife', '6.46hourglass', '6.47hammer', '6.48telephone', '6.49binoculars', '6.5barbell']

# aa = ['1.1hotdog', '1.2bread', '1.3burger', '1.4lemon', '1.5squash', '1.6mushroom', '1.7apple_2', '1.8simit', '1.9hotdog_2', '1.10burger_2', '2.1car_2', '2.2tank', '2.3wheel', '2.4wheel_2', '2.5scooter', '2.6forklift', '2.7bike', '2.8traffic_light', '2.9warplane', '2.10ship', '3.1sofa_2', '3.2crate', '3.3toaster', '3.4chair', '3.5bathing', '3.6sofa_3', '3.7piano', '3.8chair_2', '3.9washer', '3.10vase', '4.1mic', '4.2computer', '4.3mouse', '4.4camera', '4.5control', '4.6lamp', '4.7guitar', '4.8iron', '4.9keyboard', '4.10type', '5.1trash_bin', '5.2bottle', '5.3shoe', '5.4basketball', '5.5shoe_2', '5.6pool_table', '5.7sign', '5.8sign_2', '5.9sign_3', '5.10manhole', '6.1clock', '6.2clock_2', '6.3warplane_2', '6.4chair_3', '6.5car_3', '6.6sofa_4', '6.7glass', '6.8phone', '6.9guitar_2', '6.10volleyball', '6.11barrel', '6.12mic_2', '6.13scooter_2', '6.14tool', '6.15shoe_3', '6.16candle', '6.17coffeepot', '6.18vase_2', '6.19teapot', '6.20plane', '6.21car_4', '6.22soap', '6.23ipod', '6.24cassette', '6.25sofa_5', '6.26lip', '6.27sax', '6.28cart', '6.29cart_2', '6.30barrel_2', '6.31light', '6.32desk', '6.33coffee', '6.34cup', '6.35goblet', '6.36umbrella', '6.37helmet', '6.38syringe', '6.39swing', '6.40teddy', '6.41screwdriver', '6.42bank', '6.43lock', '6.44abacus', '6.45knife', '6.46hourglass', '6.47hammer', '6.48telephone', '6.49binoculars', '6.50barbell']


bb = ['hotdog, hot dog, red hot', 'French loaf', 'cheeseburger', 'lemon', 'butternut squash', 'agaric', 'Granny Smith', 'pretzel', 'hotdog, hot dog, red hot', 'cheeseburger', 'sports car, sport car', 'tank, army tank, armored combat vehicle, armoured combat vehicle', 'car wheel', 'car wheel', 'motor scooter, scooter', 'forklift', 'mountain bike, all-terrain bike, off-roader', 'traffic light, traffic signal, stoplight', 'warplane, military plane', 'canoe', 'studio couch, day bed', 'crate', 'toaster', 'rocking chair, rocker', 'washbasin, handbasin, washbowl, lavabo, wash-hand basin', 'studio couch, day bed', 'grand piano, grand', 'folding chair', 'washer, automatic washer, washing machine', 'vase', 'microphone, mike', 'notebook, notebook computer', 'mouse, computer mouse', 'reflex camera', 'remote control, remote', 'lampshade, lamp shade', 'electric guitar', 'iron, smoothing iron', 'computer keyboard, keypad', 'typewriter keyboard', 'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin', 'water bottle', 'running shoe', 'basketball', 'sandal', 'pool table, billiard table, snooker table', 'street sign', 'street sign', 'street sign', 'manhole cover', 'analog clock', 'digital clock', 'warplane, military plane', 'folding chair', 'sports car, sport car', 'studio couch, day bed', 'sunglass', 'cellular telephone, cellular phone, cellphone, cell, mobile phone', 'acoustic guitar', 'volleyball', 'barrel, cask', 'microphone, mike', 'moped', "carpenter's kit, tool kit", 'clog, geta, patten, sabot', 'candle, taper, wax light', 'coffeepot', 'vase', 'teapot', 'airliner', 'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon', 'soap dispenser', 'iPod', 'cassette', 'studio couch, day bed', 'lipstick, lip rouge', 'sax, saxophone', 'barrow, garden cart, lawn cart, wheelbarrow', 'shopping cart', 'barrel, cask', 'spotlight, spot', 'desk', 'espresso', 'cup', 'goblet', 'umbrella', 'crash helmet', 'syringe', 'swing', 'teddy, teddy bear', 'screwdriver', 'piggy bank, penny bank', 'padlock', 'abacus', 'letter opener, paper knife, paperknife', 'hourglass', 'hammer', 'dial telephone, dial phone', 'binoculars, field glasses, opera glasses', 'dumbbell']

# aa = ['bread','bus','keyboard','orange','pinapple','plane_2','plane_3','shoe']

# bb = ['French loaf','school bus','computer keyboard, keypad','orange','pineapple, ananas','warplane, military plane','warplane, military plane','running shoe']



accs = []
for i in range(0,100):
    # path_1 = "/HOME/scz1972/run/rsw_/NeRFAttack/NeRF/phyattack_real_datta/phyattack_real_datta/" + aa[i] +"/"
    path_1 = "/HOME/scz1972/run/rsw_/NeRFAttack/NeRF/train_data_all/" + aa[i] +"/train/"
    label_1 = bb[i]
    
    
    # models_k = ['inception_resnet']
    # models = [ 'inc-v3','inception_resnet','densenet','efficientnet','mlp-mixer','deit_base','swin_base','VGG','mobilenet-v2']
    models = []

    models.append(opt.model)
# models = ['resnet50', 'resnet101', 'inc-v3', 'googlenet', 'shufflenet', 'mobilnet', 'resnext', 'vit', 'resnet50-augmix', 'resnet50-deepaug', 'resnet50-deepaug+augmix']
#     models_2 = ['inc-v4', 'inception_resnet', 'VGG', 'mobilenet-v2', 'squeeze', 'mnasnet', 'deit_base', 'swin_base']
#     models_3 = ['vit-mae']

    # models_4 = ['resnet50', 'vit']
    mode = 'test_nerf_data_black_attack'
# mode = 0
    if mode == 'test_nerf_data_black_attack':
        
        for model in models:
            acc = test_baseline (path=path_1, label=label_1, model=model)
            # acc = test_baseline (path="/HOME/scz1972/run/rsw_/NeRFAttack/NeRF/nerf_blackbox_data/resnet_AP_lamba0/apple_2/", label='Granny Smith', model=model)
            accs.append(acc)
print(accs)


