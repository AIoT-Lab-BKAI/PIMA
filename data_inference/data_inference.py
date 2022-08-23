import json
import string
import os.path as osp
import networkx as nx
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
from torch_geometric.data import Dataset
from torch_geometric.utils.convert import from_networkx
from genericpath import isfile
import config as CFG
from transformers import AutoTokenizer
import matplotlib.pyplot as plt


class PrescriptionPillData(Dataset):
    def __init__(self, json_files, args):
        self.args = args
        self.text_sentences_tokenizer = AutoTokenizer.from_pretrained(
            args.text_model_name)
        self.json_files = json_files
        self.transforms = get_transforms(args.image_size)

    def create_graph(self, bboxes, imgw, imgh):
        G = nx.Graph()
        for src_idx, src_row in enumerate(bboxes):
            src_row["x_min"], src_row["y_min"], src_row["x_max"], src_row["y_max"] = src_row["box"]
            src_row['bbox'] = list(map(float, [
                                   src_row["x_min"] / imgw,
                                   src_row["y_min"] / imgh,
                                   src_row["x_max"] / imgw,
                                   src_row["y_max"] / imgh
                                   ]))

            G.add_node(
                src_idx,
                text=src_row['text'],
                bbox=src_row['bbox'],
            )

            src_range_x = (src_row["x_min"], src_row["x_max"])
            src_range_y = (src_row["y_min"], src_row["y_max"])

            # Xác định hàng xóm lân cận
            neighbor_vert_bot = []
            neighbor_hozi_right = []

            for dest_idx, dest_row in enumerate(bboxes):
                if dest_idx == src_idx:
                    continue
                dest_row["x_min"], dest_row["y_min"], dest_row["x_max"], dest_row["y_max"] = dest_row["box"]
                dest_range_x = (dest_row["x_min"], dest_row["x_max"])
                dest_range_y = (dest_row["y_min"], dest_row["y_max"])

                # Find box in horizontal must have common x range.
                if max(src_range_x[0], dest_range_x[0]) < min(src_range_x[1], dest_range_x[1]):
                    # Find underneath box: neighbor yminx must be smaller than source ymax
                    if dest_range_y[0] >= src_range_y[1]:
                        neighbor_vert_bot.append(dest_idx)

                # Find box in horizontal must have common y range.
                if max(src_range_y[0], dest_range_y[0]) < min(src_range_y[1], dest_range_y[1]):
                    # Find right box: neighbor xmin must be smaller than source xmax
                    if dest_range_x[0] >= src_range_x[1]:
                        neighbor_hozi_right.append(dest_idx)

            neighbors = []
            if neighbor_hozi_right:
                nei = min(neighbor_hozi_right,
                          key=lambda x: bboxes[x]['x_min'])
                neighbors.append(nei)
                G.add_edge(src_idx, nei)

            if neighbor_vert_bot:
                nei = min(neighbor_vert_bot, key=lambda x: bboxes[x]['y_min'])
                neighbors.append(nei)
                G.add_edge(src_idx, nei)

        return G

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(self.json_files[idx], str):
            with open(self.json_files[idx], "r") as f:
                prescription = json.load(f)

        # FOR IMAGE PILLS        
        pills_image_path = self.args.data_folder + self.args.image_path
        pills_image_folder = torchvision.datasets.ImageFolder(pills_image_path, transform=self.transforms)

        pills_loader = torch.utils.data.DataLoader(pills_image_folder, batch_size=32, shuffle=False)

        pills_images = []
        pills_labels = []
        for images, labels in pills_loader:
            pills_images.append(images)
            pills_labels.append(labels)
        
        # FOR PRESCRIPTIONS
        G = self.create_graph(bboxes=prescription, imgw=2000, imgh=2000)
        data = from_networkx(G)

        text_sentences = self.text_sentences_tokenizer(
            data.text, max_length=32, padding='max_length', truncation=True, return_tensors='pt')
        data.text_sentences_ids, data.text_sentences_mask = text_sentences.input_ids, text_sentences.attention_mask

        data.pills_images = pills_images[0]
        data.pills_labels = pills_labels[0]
        return data


def get_transforms(size=224):
    transform = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform


def get_all_pill_label(pills_image_path):
    pills_image_folder = torchvision.datasets.ImageFolder(pills_image_path)
    return pills_image_folder.class_to_idx
