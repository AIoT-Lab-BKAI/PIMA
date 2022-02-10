from data.data import PrescriptionPillData
from torch_geometric.data import DataLoader
import torch


def build_loaders(files, mode="train", batch_size=1):
    dataset = PrescriptionPillData(files, mode)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True)
    return dataloader


def creat_batch_triplet(image_embedding, graph_embedding_pills, graph_embedding_pills_labels, labels):
    anchor, positive, negative = torch.tensor([]).cuda(
    ), torch.tensor([]).cuda(), torch.tensor([]).cuda()

    for idx, label in enumerate(labels):
        positive_idx = graph_embedding_pills_labels.eq(label)
        negative_idx = graph_embedding_pills_labels.ne(label)

        anchor = torch.cat(
            (anchor, image_embedding[idx].unsqueeze(0).unsqueeze(0)))
        positive = torch.cat(
            (positive, graph_embedding_pills[positive_idx].unsqueeze(0)))

        if sum(negative_idx) == 0:
            negative = torch.cat((negative, torch.zeros_like(
                image_embedding[idx]).unsqueeze(0).unsqueeze(0)))
        else:
            negative = torch.cat(
                (negative, graph_embedding_pills[negative_idx].unsqueeze(0)))

    return anchor, positive, negative
