import torch
from torch import nn
import torch.nn.functional as F

import config as CFG
from utils import LABELS
from models import ImageEncoder, ProjectionHead, BERTxSAGE

class PrescriptionPill(nn.Module):
    def __init__(self, temperature=CFG.temperature, image_embedding=CFG.image_embedding, graph_embedding=CFG.graph_embedding, n_classes=len(LABELS)):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.graph_encoder = BERTxSAGE()
        
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.graph_projection = ProjectionHead(embedding_dim=graph_embedding)

        self.temperature = temperature
    
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

        # Calculating the Loss of Image and Graph
        logits = (graph_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        graph_similarity = graph_embeddings @ graph_embeddings.T
        targets = F.softmax(
            (images_similarity + graph_similarity) / 2 * self.temperature, dim=-1
        )
        
        images_loss = cross_entropy(logits, targets, reduction='none') 
        graph_loss = cross_entropy(logits.T, targets.T, reduction='none') 
        loss_graph_images = (images_loss + graph_loss) / 2.0

        # For graph KIE 
        graph_extract = self.post_process_layers(graph_features)
        graph_extract = F.log_softmax(graph_extract, dim=1)

        return graph_extract, loss_graph_images.mean()

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
