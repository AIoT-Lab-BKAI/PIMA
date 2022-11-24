# PIMA - A Novel Approach for Pill-Prescription Matching with GNN Assistance and Contrastive Learning

This repository is an implementation of "A Novel Approach for Pill-Prescription Matching with GNN Assistance and Contrastive Learning" by
Trung Thanh Nguyen, Hoang Dang Nguyen, Thanh Hung Nguyen, Huy Hieu Pham, Ichiro Ide, and Phi Le Nguyen" by Huy-Hieu Pham. The paper was accepted for presentation at The 19th Pacific Rim International Conference on Artificial Intelligence Shanghai, China November 10-13, 2022.

Full paper is available [here](https://github.com/AIoT-Lab-BKAI/PIMA/tree/main/paper).

---
Environment setting using [Anaconda](https://www.anaconda.com/).

```
conda create --name pima
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pyg -c pyg
conda install -c conda-forge transformers
conda install -c conda-forge timm
conda install -c anaconda networkx
conda install -c conda-forge wandb
```
