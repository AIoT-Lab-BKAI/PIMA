from torch import nn
import torch.nn.functional as F

import config as CFG
from utils import LABELS
from models import ImageEncoder, ProjectionHead, BERTxSAGE

class PrescriptionPill(nn.Module):
    def __init__(self, image_embedding=CFG.image_embedding, graph_embedding=CFG.graph_embedding, n_classes=len(LABELS)):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.graph_encoder = BERTxSAGE()
        
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.graph_projection = ProjectionHead(embedding_dim=graph_embedding)

        self.post_process_layers = nn.Sequential(
            nn.BatchNorm1d(256, affine= False),
            nn.Dropout(p=0.2),
            nn.Linear(256, n_classes),
            nn.GELU()
        )

    def forward(self, data):
        # Getting graph embedding
        graph_features = self.graph_encoder(data)
        image_features = self.image_encoder(data.img)

        # Getting Image and Graph Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        graph_embeddings = self.graph_projection(graph_features)

        # For graph KIE 
        graph_extract = self.post_process_layers(graph_features)
        graph_extract = F.log_softmax(graph_extract, dim=1)

        return graph_extract, image_embeddings, graph_embeddings

