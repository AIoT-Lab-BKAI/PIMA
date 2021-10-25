import torch
from sklearn.metrics import classification_report

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
