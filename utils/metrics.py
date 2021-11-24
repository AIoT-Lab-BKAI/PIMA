import torch
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import config as CFG
import torch.nn.functional as F
from torch import nn

class MetricTracker:
    def __init__(self, labels=None):
        super().__init__()
        self.target_names = labels
        self.preds = []
        self.targets = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = torch.cat(self.preds).cpu().numpy()
        targets = torch.cat(self.targets).cpu().numpy()
        return classification_report(targets, preds,
                                target_names=self.target_names, zero_division=0)

    def reset(self):
        self.preds = []
        self.targets = []

class MatchingMetric: 
    def __init__(self, temperature=CFG.temperature):
        super().__init__()
        self.temperature = temperature

    def compute_loss(self, image_embeddings, graph_embeddings, matching_label):
        # image_embeddings = image_embeddings[matching_label == 1]

        logits = (image_embeddings @ graph_embeddings.T) / self.temperature
        # logits = (graph_embeddings @ image_embeddings.T) / self.temperature

        images_similarity = image_embeddings @ image_embeddings.T
        graph_similarity = graph_embeddings @ graph_embeddings.T

        targets = F.softmax(
            (images_similarity + graph_similarity) / 2 * self.temperature, dim=-1
        )

        images_loss = self.cross_entropy(logits, targets, reduction='none') 
        graph_loss = self.cross_entropy(logits.T, targets.T, reduction='none') 
        pill_prescription_loss = (images_loss + graph_loss) / 2.0

        return pill_prescription_loss.mean()

    def cross_entropy(self, logits, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(logits)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
    
    def accuracy(self, image_embeddings, graph_embeddings, matching_label):
        image_embeddings = image_embeddings[matching_label == 1]
        logits = (image_embeddings @ graph_embeddings.T) / self.temperature

        preds = logits.argmax(dim=1)
        targets = [i for i, v in enumerate(matching_label) if v == 1]

        accuracy = accuracy_score(targets, preds.cpu())
        
        return accuracy
