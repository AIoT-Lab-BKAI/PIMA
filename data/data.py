import json
import string
import os.path as osp
import networkx as nx
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Dataset
from torch_geometric.utils.convert import from_networkx
from transformers import RobertaTokenizer
from genericpath import isfile

import albumentations as A
import cv2

from utils import LABELS
import config as CFG


class PrescriptionPillData(Dataset):
    def __init__(self, json_files, mode, bert_model="roberta-base"):
        """
        Args:
            json_files: list of label json file paths
        """
        self.tokenizer = RobertaTokenizer.from_pretrained(bert_model)
        self.json_files = json_files
        self.mode = mode
        self.transforms = get_transforms()

    def connect(self, bboxes, imgw, imgh, img_folder_name):
        G = nx.Graph()
        for src_idx, src_row in enumerate(bboxes):
            src_row['label'] = src_row['label'].lower()

            if not src_row['label']:
                src_row['label'] = "other"
            
            img = torch.zeros_like(torch.empty(CFG.depth, CFG.size, CFG.size))
            if src_row['label'] == 'drugname':
                # TODO: Update for multi load image
                # Get only 1 file
                img_files = src_row['img'][0]
                image = cv2.imread(f"{CFG.image_path}/{self.mode}/{img_folder_name}/{img_files}")
                image = self.transforms(image=image)['image']
                img = torch.tensor(image).permute(2, 0, 1).float()

            src_row['y'] = torch.tensor([LABELS.index(src_row['label'])], dtype=torch.long)

            src_row["x_min"], src_row["y_min"], src_row["x_max"], src_row["y_max"] = src_row["box"]
            src_row['bbox'] = list(map(float, [
                                   src_row["x_min"]/imgw, 
                                   src_row["y_min"]/imgh, 
                                   src_row["x_max"]/imgw, 
                                   src_row["y_max"]/imgh
                                ]))
            

            if not len(src_row['text']):
                p_num = 0.0
            else:
                p_num = sum([n in string.digits for n in src_row['text']]) / len(src_row['text'])

            G.add_node(
                src_idx,
                text=src_row['text'],
                bbox=src_row['bbox'],
                label=src_row['label'],
                p_num=p_num,
                y=src_row['y'],
                img=img
               )

            src_range_x = (src_row["x_min"], src_row["x_max"])
            src_range_y = (src_row["y_min"], src_row["y_max"])

            src_center_x, src_center_y = np.mean(
                src_range_x), np.mean(src_range_y)

            neighbor_vert_top = []
            neighbor_vert_bot = []
            neighbor_hozi_left = []
            neighbor_hozi_right = []

            for dest_idx, dest_row in enumerate(bboxes):
                if dest_idx == src_idx:
                    continue
                dest_row["x_min"], dest_row["y_min"], dest_row["x_max"], dest_row["y_max"] = dest_row["box"]
                dest_range_x = (dest_row["x_min"], dest_row["x_max"])
                dest_range_y = (dest_row["y_min"], dest_row["y_max"])
                dest_center_x, dest_center_y = np.mean(
                    dest_range_x), np.mean(dest_range_y)
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

            if neighbor_hozi_left:
                nei = max(neighbor_hozi_left, key=lambda x: bboxes[x]['x_max'])
                neighbors.append(nei)
                G.add_edge(src_idx, nei)
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
                raw = json.load(f)
                f.close()
                filename = str(self.json_files[idx]).split('/')[-1].split('.')[0]
        else:
            # TODO: CHECK IT LATER 
            # raw = self.json_files[idx]["anno"]
            pass

        G = self.connect(bboxes=raw,
                         imgw=2000, imgh=2000, img_folder_name=filename)
        
        # For Draw Graph IMG 
        # nx.draw(G,node_size= 20, with_labels = True)
        # if not isfile('./vis/image_'+str(idx)+'.png'):
            # plt.savefig('./vis/image_'+str(idx)+'.png')

        data = from_networkx(G)
        token = self.tokenizer(data.text, add_special_tokens=True, truncation=True,
                               max_length=128, padding='max_length', return_tensors='pt')
        data.input_ids, data.attention_mask = token.input_ids, token.attention_mask
        data.text_len = torch.count_nonzero(data.input_ids, dim=1) / 128.0
        data.text_len = torch.unsqueeze(data.text_len, dim=1)

        data.bbox = torch.Tensor(data.bbox)
        data.p_num = torch.Tensor(data.p_num)
        data.p_num = torch.unsqueeze(data.p_num, dim=1)

        data.img = torch.stack(data.img)

        if isinstance(self.json_files[idx], str):
            data.path = self.json_files[idx]
            data.imname = osp.basename(data.path)
        else:
            # TODO: CHECK IT LATER 
            # data.imname = self.json_files[idx]["fname"]
            pass
        return data

def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
