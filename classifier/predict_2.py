from sympy import true
import torch
from torchvision import models
from torchvision import transforms
import cv2
from PIL import Image
import math
import numpy as np
from zmq import device
import os
import timm
# import matplotlib.pyplot as plt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(dir(models))


def test_baseline(path, label, model, is_mean=False):

    images_path = path

    images_name = []
    fileList=os.listdir(images_path)
    n = 0
    for i in fileList:
        images_name.append(fileList[n])
        n = n+1
    images_data = [] # opencv
    tensor_data = [] # pytorch tensor

    for name in images_name:
        print('name:')
        print(name)
        if is_mean:
            if name == '100.png':
                img = cv2.imread(images_path + name)
            else:
                continue
        else:
            img = cv2.imread(images_path + name)
        print(f"name: {images_path+name}, opencv image shape: {img.shape}") # (h,w,c)
        images_data.append(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)

        transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        tensor = transform(img_pil)
        #print(f"tensor shape: {tensor.shape}, max: {torch.max(tensor)}, min: {torch.min(tensor)}") # (c,h,w)
        tensor = torch.unsqueeze(tensor, 0) # 返回一个新的tensor,对输入的既定位置插入维度1
        #print(f"tensor shape: {tensor.shape}, max: {torch.max(tensor)}, min: {torch.min(tensor)}") # (1,c,h,w)
        tensor_data.append(tensor)

    # if model == 'resnet50':
    #     model = models.resnet50(pretrained=True)
    # if model == 'resnet101':
    #     model = models.resnet101(pretrained=True)
    # if model == 'inc-v3':
    #     model = models.inception_v3(pretrained=True)
    # if model == 'googlenet':
    #     model = models.googlenet(pretrained=True)
    # if model == 'shufflenet':
    #     model = models.shufflenet_v2_x0_5(pretrained=True)
    # if model == 'mobilnet':
    #     model = models.mobilenet_v3_large(pretrained=True)
    # if model == 'resnext':
    #     model = models.resnext50_32x4d(pretrained=True)
    # if model == 'vit':
    #     model = models.vit_b_16(pretrained=True)


    if model == 'resnet50-augmix':
        model = models.resnet50(pretrained=False)
        # model = models.__dict__['resnet50']
        checkpoint = torch.load('C:/Users/Silvester/.cache/torch/hub/checkpoints/augmix.pth.tar')
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(checkpoint['state_dict'])

    if model == 'resnet50-deepaug':
        model = models.resnet50(pretrained=False)
        # model = models.__dict__['resnet50']
        checkpoint = torch.load('C:/Users/Silvester/.cache/torch/hub/checkpoints/deepaugment.pth.tar')
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(checkpoint['state_dict'])

    if model == 'resnet50-deepaug+augmix':
        model = models.resnet50(pretrained=False)
        # model = models.__dict__['resnet50']
        checkpoint = torch.load('C:/Users/Silvester/.cache/torch/hub/checkpoints/deepaugment_and_augmix.pth.tar')
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(checkpoint['state_dict'])

    # if model == 'VGG':
    #     model = models.vgg16(pretrained=True)
    # if model == 'mobilenet-v2':
    #     model = models.mobilenet_v2(pretrained=True)
    if model == 'squeeze':
        model = models.squeezenet1_0(pretrained=True)
    if model == 'mnasnet':
        model = models.mnasnet1_0(pretrained=True)

    # 利用timm
    if model == 'inc-v3':
        model = timm.create_model('inception_v3', pretrained=True)
    if model == 'inception_resnet':
        model = timm.create_model('inception_resnet_v2', pretrained=True)

    if model == 'densenet':
        model = timm.create_model('densenet121', pretrained=True)

    if model == 'efficientnet':
        model = timm.create_model('efficientnet_b0', pretrained=True)

    if model == 'mlp-mixer':
        model = timm.create_model('mixer_b16_224', pretrained=True)

    if model == 'deit_base':
        model = timm.create_model('deit_base_distilled_patch16_224', pretrained=True)

    
        # checkpoint = torch.load('C:/Users/Silvester/.cache/torch/hub/checkpoints/deit_base_patch16_224-b5f2ef4d.pth')
        # # model = torch.nn.DataParallel(model).cuda()
        # model.load_state_dict(checkpoint['model'])

    if model == 'swin_base':
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        # checkpoint = torch.load('C:/Users/Silvester/.cache/torch/hub/checkpoints/swin_base_patch4_window7_224.pth')
        # # model = torch.nn.DataParallel(model).cuda()
        # model.load_state_dict(checkpoint['model'])

    if model == 'VGG':
        model = timm.create_model('vgg16', pretrained=True)
    if model == 'mobilenet-v2':
        model = timm.create_model('mobilenetv2_120d', pretrained=True)

    # 第二批：
    if model == 'vgg19':
        model = timm.create_model('vgg19', pretrained=True)
    if model == 'densenet169':
        model = timm.create_model('densenet169', pretrained=True)
    if model == 'densenet201':
        model = timm.create_model('densenet201', pretrained=True)
    if model == 'inception_v4':
        model = timm.create_model('inception_v4', pretrained=True)
    if model == 'resnet18':
        model = timm.create_model('resnet18', pretrained=True)
    if model == 'resnet34':
        model = timm.create_model('resnet34', pretrained=True)
    if model == 'resnet50':
        model = timm.create_model('resnet50', pretrained=True)
    if model == 'resnet101':
        model = timm.create_model('resnet101', pretrained=True)
    if model == 'resnet152':
        model = timm.create_model('resnet152', pretrained=True)
    if model == 'efficientnet_b1':
        model = timm.create_model('efficientnet_b1', pretrained=True)
    if model == 'efficientnet_b2':
        model = timm.create_model('efficientnet_b2', pretrained=True)
    if model == 'efficientnet_b3':
        model = timm.create_model('efficientnet_b3', pretrained=True)
    if model == 'efficientnet_b4':
        model = timm.create_model('efficientnet_b4', pretrained=True)
    if model == 'mobilenetv2_140':
        model = timm.create_model('mobilenetv2_140', pretrained=True)
    if model == 'mixer_l16_224':
        model = timm.create_model('mixer_l16_224', pretrained=True)
    if model == 'vit_base_patch16_224':
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
    if model == 'vit_large_patch16_224':
        model = timm.create_model('vit_large_patch16_224', pretrained=True)
    if model == 'deit_base_patch16_224':
        model = timm.create_model('deit_base_patch16_224', pretrained=True)
    if model == 'deit_small_patch16_224':
        model = timm.create_model('deit_small_patch16_224', pretrained=True)
    if model == 'deit_tiny_patch16_224':
        model = timm.create_model('deit_tiny_patch16_224', pretrained=True)
    if model == 'swin_large_patch4_window7_224':
        model = timm.create_model('swin_large_patch4_window7_224', pretrained=True)
    if model == 'swin_small_patch4_window7_224':
        model = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
    if model == 'swin_tiny_patch4_window7_224':
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)






    if model == 'vit-mae':
        from mae.models_mae import mae_vit_base_patch16_dec512d8b
        from mae.models_vit import vit_base_patch16
        # model = models_vit.__dict__['vit_large_patch16']()
        model = mae_vit_base_patch16_dec512d8b()
        # model = timm.create_model('deit_base_patch16_224')
        # model = torch.nn.DataParallel(model)
        checkpoint = torch.load('C:/Users/Silvester/.cache/torch/hub/checkpoints/mae_pretrain_vit_base_full.pth')
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                del checkpoint_model[k]

        model.load_state_dict(checkpoint_model)
        print(model)





    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    with open("/HOME/scz1972/run/rsw_/NeRFAttack/classifier/imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]




    acc = 0

    # for x in range(len(tensor_data)):
    #     prediction = resnet50(tensor_data[x])
    #     #print(prediction.shape) # [1,1000]

    #     _, index = torch.max(prediction, 1)
    #     percentage = torch.nn.functional.softmax(prediction, dim=1)[0] * 100
    #     print(f"result: {classes[index[0]]}, {percentage[index[0]].item()}")

    #     if classes[index[0]] == 'microphone, mike':
    #         acc += 1

    # acc = acc/len(tensor_data)
    # print("acc:", acc)


    with torch.no_grad():
        for x in range(len(tensor_data)):

            top_num = 1
            prediction = model(tensor_data[x])
            ps = torch.exp(prediction)
            topk, topclass = ps.topk(top_num, dim=1)
            class_ = []
            for i in range(top_num):
                class_.append(classes[topclass.cpu().numpy()[0][i]])
            print("Output class : ", class_)

            #true_label = np.zeros((1, 1000))
            #true_label[:, 817] = 1.0
            #true_label = torch.from_numpy(true_label)
            #loss_func = torch.nn.CrossEntropyLoss()

            #print('loss:', loss_func(prediction, true_label.to(device)))
            print('score:', np.max(ps.cpu().numpy())/np.sum(ps.cpu().numpy()))

            for i in range(len(topclass.cpu().numpy()[0])):
                if classes[topclass.cpu().numpy()[0][i]] == label:
                    acc += 1

    acc = acc/len(tensor_data)
    print("acc:", acc)
    return acc
        