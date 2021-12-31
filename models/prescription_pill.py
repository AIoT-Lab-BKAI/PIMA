import torch
from torch import nn
import torch.nn.functional as F

import config as CFG
from models import ImageEncoder, ProjectionHead, BERTxSAGE, ImageEncoderTimm, sentencesTransformer, SBERTxSAGE


class PrescriptionPill(nn.Module):
    def __init__(self, image_embedding=CFG.image_embedding, text_embedding = CFG.text_embedding, graph_embedding=CFG.graph_embedding, graph_n_classes=len(CFG.LABELS), image_pretrained_link=CFG.image_pretrained_link):
        super().__init__()
        # self.image_encoder = ImageEncoderTimm()
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

        # self.text_encoder = TextEncoder()
        self.graph_encoder = SBERTxSAGE()
        self.graph_sentences_encoder = sentencesTransformer()

        self.sentences_encoder = sentencesTransformer()

        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.graph_projection = ProjectionHead(embedding_dim=graph_embedding)

    #     self.post_process_layers = nn.Sequential(
    #         nn.BatchNorm1d(256, affine=False),
    #         nn.Dropout(p=0.2),
    #         nn.Linear(256, graph_n_classes),
    #         nn.GELU()
    #     )

    # def forward_graph(self, data):
    #     # Getting graph embedding
    #     graph_features = self.graph_encoder(data, self.sentences_encoder(data.text_sentences_ids, data.text_sentences_mask))
    #     # FOR KIE 
    #     graph_extract = self.post_process_layers(graph_features)
    #     graph_extract = F.log_softmax(graph_extract, dim=-1)
    #     return graph_extract

    def get_image_features(self, image):
        x = self.image_encoder(image)
        x = self.image_projection(x)
        return x

    def get_graph_features(self, data):
        x = self.graph_encoder(data, self.graph_sentences_encoder(data.text_sentences_ids, data.text_sentences_mask))
        x = self.graph_projection(x)
        return x

    def get_sentences_features(self, data):
        x = self.sentences_encoder(data.text_sentences_ids, data.text_sentences_mask)
        x = self.text_projection(x)
        return x 

    def forward(self, data, pills_image):
        return self.get_image_features(pills_image), self.get_sentences_features(data)

    def graph_forward(self, data, pills_image):
        return self.get_image_features(pills_image), self.get_sentences_features(data), self.get_graph_features(data)
