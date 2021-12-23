
import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import argparse
import os
from model import ImageEncoder
from torch import nn
import numpy as np 
from sklearn.model_selection import train_test_split

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Network Inference')
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--pretrain_path', default='/mnt/disk1/vaipe-thanhnt/EMED-Prescription-and-Pill-matching/pills-resnet/logs/model_0.pth')
    parser.add_argument('--image_path', default='/mnt/disk1/vaipe-data/pills/result/pill-cropped-211012/')
    
    args = parser.parse_args()
    return args 


def inference(args):
    print('==> Preparing data..')
    # Setting dataset
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    data_test = torchvision.datasets.ImageFolder(args.image_path, transform=transform)
    data_loader_test = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=True, num_workers=4)


    print(data_test[2][0])
    print(data_test.targets)
    targets = data_test.targets
    # tran test split targets

    need, _ = train_test_split(targets, test_size=0.99, random_state=42)
    print(len(need))



    # print('Number of test images:', len(data_test))
    # _, dict_class = data_test._find_classes(args.image_path)
    # print(dict_class)
    # print(dict_class['Zelfamox-875-125-DT'])

    # print(data_test.class_to_idx)

    # get image in data_test
    # for i in range(len(data_test)):
        # print(data_test.imgs[i])
        # break
    # dataset_subset = torch.utils.data.Subset(data_test, np.random.choice(len(data_test), 10, replace=False)


    # for data, target in data_loader_test:
        # print(data.shape)
        # print(target)
        # break

    # print('==> Building model..')
    # model = ImageEncoder()

    # # create image 
    # img = torch.randn(1, 3, 224, 224)

    # # load pretrained model
    # model_state_dict = model.state_dict()
    # print(model(img))

    # pretrained_dict = torch.load(args.pretrain_path)

    # # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_state_dict}
    # # 2. overwrite entries in the existing state dict
    # model_state_dict.update(pretrained_dict)
    # # 3. load the new state dict
    # model.load_state_dict(model_state_dict)

    # print(model(img).shape)

if __name__ == "__main__":
    args = parse_args()
    inference(args)

