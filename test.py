
# Check CUDA Visiable 
import torch
print(torch.cuda.is_available())
from torch import nn
import cv2
import config as CFG
import albumentations as A
import timm

# def get_transforms(mode="train"):
#     if mode == "train":
#         return A.Compose(
#             [
#                 A.Resize(CFG.size, CFG.size, always_apply=True),
#                 A.Normalize(max_pixel_value=255.0, always_apply=True),
#             ]
#         )
#     else:
#         return A.Compose(
#             [
#                 A.Resize(CFG.size, CFG.size, always_apply=True),
#                 A.Normalize(max_pixel_value=255.0, always_apply=True),
#             ]
#         )

# image = cv2.imread("/mnt/disk1/vaipe-thanhnt/EMED-Prescription-and-Pill-matching/data/pills/train/01/1.png")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # print(image)
# transforms = get_transforms(mode="train")
# image = transforms(image=image)['image']
# item = torch.Tensor(image).permute(2, 0, 1).float()
# # print(item)

# class ImageEncoder(nn.Module):
#     """
#     Encode images to a fixed size vector
#     """

#     def __init__(
#         self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
#     ):
#         super().__init__()
#         self.model = timm.create_model(
#             model_name, pretrained, num_classes=0, global_pool="avg"
#         )
#         for p in self.model.parameters():
#             p.requires_grad = trainable

#     def forward(self, x):
#         return self.model(x)


# # image_encoder = ImageEncoder()
# # forward = image_encoder(item)

input = torch.empty(2, 3)

print(torch.zeros_like(torch.empty(2, 3)))
