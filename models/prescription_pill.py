import torch
from torch import nn
import torch.nn.functional as F

import config as CFG
from models import ImageEncoder, ProjectionHead, BERTxSAGE


class PrescriptionPill(nn.Module):
    def __init__(self, image_embedding=CFG.image_embedding, graph_embedding=CFG.graph_embedding, graph_n_classes=len(CFG.LABELS), image_pretrained_link=CFG.image_pretrained_link):
        super().__init__()
        self.image_encoder = ImageEncoder()
        if image_pretrained_link is not None:
            pre_train_state = torch.load(image_pretrained_link)
            image_model_state = self.image_encoder.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pre_train_state.items() if k in image_model_state}
            # 2. overwrite entries in the existing state dict
            image_model_state.update(pretrained_dict)
            # 3. load the new state dict
            self.image_encoder.load_state_dict(image_model_state)
            print("Loaded pretrained image model successfully!")

        self.graph_encoder = BERTxSAGE()

        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.graph_projection = ProjectionHead(embedding_dim=graph_embedding)

        self.post_process_layers = nn.Sequential(
            nn.BatchNorm1d(256, affine=False),
            nn.Dropout(p=0.2),
            nn.Linear(256, graph_n_classes),
            nn.GELU()
        )

    def forward(self, data, pills_image):
        # Getting graph embedding
        graph_features = self.graph_encoder(data)
        image_features = self.image_encoder(pills_image)
        
        # if mode == "train":
        #     graph_features = graph_features[data.pills_label >= 0]
        #     new_label = data.pills_label[data.pills_label >= 0]

        #     # sort by label
        #     graph_features = graph_features[new_label.argsort()]

        image_embedding = self.image_projection(image_features)
        graph_embedding = self.graph_projection(graph_features)

        return image_embedding, graph_embedding


