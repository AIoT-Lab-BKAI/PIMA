
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import os
from model import ImageEncoder

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--epochs', default=10)
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--seed', default=42)
    parser.add_argument('--save_path', default='/mnt/disk1/vaipe-thanhnt/EMED-Prescription-and-Pill-matching/pills-resnet/logs/')
    parser.add_argument('--image_path', default='/mnt/disk1/vaipe-data/pills/result/pill-cropped-211012/')
    
    args = parser.parse_args()
    return args 

def spliting(source_path):
    return source_path, source_path

def main(args):
    print('==> Preparing data..')
    # Setting dataset
    train_path, val_path = spliting(args.image_path)
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.RandomRotation(10),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    data_train = torchvision.datasets.ImageFolder(train_path, transform=transform)
    data_val = torchvision.datasets.ImageFolder(val_path, transform=transform)
    data_loader_train = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    data_loader_val = torch.utils.data.DataLoader(data_val, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print('Number of training images:', len(data_train))
    print('Number of validation images:', len(data_val))

    print('==> Building model..')
    # Setting Model
    num_classes = len(os.listdir(args.image_path))

    imageEncoderModel = ImageEncoder(pretrained=True, trainable=True, num_classes=num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imageEncoderModel.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(imageEncoderModel.parameters(), lr=args.lr, momentum=args.momentum)

    print('==> Training model..')
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(data_loader_train):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = imageEncoderModel(images)

            print(outputs)
            print(outputs.shape)
            break 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, args.epochs, i+1, len(data_loader_train), loss.item()))
        break
        # correct = 0
        # total = 0
        # with torch.no_grad():
        #     for images, labels in data_loader_val:
        #         images = images.to(device)
        #         labels = labels.to(device)
        #         outputs = imageEncoderModel(images)
        #         _, predicted = torch.max(outputs.data, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()

        # print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))
        # torch.save(imageEncoderModel.state_dict(), args.save_path + 'model_' + str(epoch) + '.pth')


if __name__ == "__main__":
    args = parse_args()
    main(args)

