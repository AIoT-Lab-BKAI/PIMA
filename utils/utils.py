from data.data import PrescriptionPillData
from torch_geometric.data import DataLoader
import torch
import math


def build_loaders(files, mode="train", batch_size=1, args=None):
    dataset = PrescriptionPillData(files, mode, args)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True)
    return dataloader


def calculate_matching_loss(image_aggregation, text_embedding_drugname, text_embedding_labels, pills_images_labels, matching_criterion, negative_ratio=None):

    loss = []
    for idx, label in enumerate(pills_images_labels):
        positive_idx = text_embedding_labels.eq(label)
        negative_idx = text_embedding_labels.ne(label)

        anchor = image_aggregation[idx]
        positive = text_embedding_drugname[positive_idx]
        negative = text_embedding_drugname[negative_idx]

        if negative_ratio is not None:
            # get random negative samples
            negative = negative[torch.randperm(
                len(negative))[:math.ceil(len(negative) * negative_ratio)]]

        loss.append(matching_criterion(anchor, positive, negative))

    return torch.mean(torch.stack(loss))
