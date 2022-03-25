from data.data import PrescriptionPillData
from torch_geometric.data import DataLoader
import torch
import math

def build_loaders(files, mode="train", batch_size=1):
    dataset = PrescriptionPillData(files, mode)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True)
    return dataloader


def creat_batch_triplet(image_features, text_embedding_drugname, text_embedding_labels, pills_images_labels):
    anchor, positive, negative = torch.tensor([]).cuda(
    ), torch.tensor([]).cuda(), torch.tensor([]).cuda()

    for idx, label in enumerate(pills_images_labels):
        positive_idx = text_embedding_labels.eq(label)
        negative_idx = text_embedding_labels.ne(label)

        anchor = torch.cat(
            (anchor, image_features[idx].unsqueeze(0).unsqueeze(0)))
        positive = torch.cat(
            (positive, text_embedding_drugname[positive_idx].unsqueeze(0)))

        if sum(negative_idx) == 0:
            negative = torch.cat((negative, torch.zeros_like(
                image_features[idx]).unsqueeze(0).unsqueeze(0)))
        else:
            negative = torch.cat(
                (negative, text_embedding_drugname[negative_idx].unsqueeze(0)))

    return anchor, positive, negative


def creat_batch_triplet_random(image_features, text_embedding_drugname, text_embedding_labels, pills_images_labels, ratio=0.2):
    anchor, positive, negative = torch.tensor([]).cuda(
    ), torch.tensor([]).cuda(), torch.tensor([]).cuda()

    for idx, label in enumerate(pills_images_labels):
        positive_idx = text_embedding_labels.eq(label)
        positive_idx = torch.nonzero(positive_idx).squeeze(1)
        negative_idx = text_embedding_labels.ne(label)
        negative_idx = torch.nonzero(negative_idx).squeeze(1)

        anchor = torch.cat(
            (anchor, image_features[idx].unsqueeze(0).unsqueeze(0)))

        positive = torch.cat(
            (positive, text_embedding_drugname[positive_idx[0]].unsqueeze(0).unsqueeze(0)))

        if len(negative_idx) == 0:
            negative = torch.cat((negative, torch.zeros_like(
                image_features[idx]).unsqueeze(0).unsqueeze(0)))
        else:
            negative_idx = torch.randperm(len(negative_idx))[
                :math.ceil(len(negative_idx) * ratio)]
            negative = torch.cat(
                (negative, text_embedding_drugname[negative_idx].unsqueeze(0)))

    # print(anchor.shape, positive.shape, negative.shape)
    return anchor, positive, negative


# create tensor size [10, 256]
# image_features = torch.randn(5, 256).cuda()
# text_embedding_drugname = torch.randn(4, 256).cuda()
# text_embedding_labels = torch.tensor([1, 1, 0, -2, -1]).cuda()
# pills_images_labels = torch.tensor([1, 0, 1, 0, 1]).cuda()
# creat_batch_triplet_random(image_features, text_embedding_drugname,
#                            text_embedding_labels, pills_images_labels, 0.2)
